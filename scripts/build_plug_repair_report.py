#!/usr/bin/env python3
"""Build a self-contained diagnosis page from measured repair artifacts."""

import hashlib
import html
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("TRAINING_REPORT_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUT = ROOT / "artifacts/plug_repair"


def read(name, default=None):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else default


def main():
    now = datetime.now(timezone.utc).isoformat()
    source_dir = OUT / "sources"
    source_dir.mkdir(exist_ok=True)
    legacy = ROOT.parent / "active_perception_arms/src/active_perception_arms/tasks/insertion"
    provenance = []
    for source, destination in (
        (legacy / "README.md", "legacy-insertion-README.md"),
        (legacy / "mdp/rewards.py", "legacy-rewards.py"),
        (ROOT / "artifacts/plug_analysis/legacy_run_inventory.json", "legacy-run-inventory.json"),
    ):
        target = source_dir / destination
        shutil.copyfile(source, target)
        provenance.append(
            {
                "source": str(source),
                "snapshot": str(target.relative_to(OUT)),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    diagnosis = read("xm-native-diagnosis.json")
    smoke = read("smoke-runtime.json", {})
    gpu = read("gpu-validation.json", {})
    training = read("train-runtime.json", {})
    submission = read("train-submission.json", {})
    plan = read("recipe-plan.json")
    status = training.get("status", submission.get("status", "prepared; not submitted"))
    job = training.get("job_id", submission.get("job_id", "—"))
    lead_title = "XM is reachable.<br>The training recipe needs a fair test."
    lead_text = "The legacy checkout exposes substantial reward, architecture, budget and task differences. A reproduced hard XM failure disappears when the controller lifts before traversing. Optional wrist yaw is implemented; learned success after repair is still being measured."
    if status == "complete":
        lead_title = "The legacy recipe and yaw<br>did not solve XM."
        lead_text = "Both learning runs and their repeated physical evaluations have finished. XYZ scored 77/512, entirely on YP; XYZ+yaw scored 0/512. All XM evaluations failed. The completed comparison below preserves the exact settings and measured outcomes."
    diagnosis_link = (
        '<p><a href="../xm_diagnosis/index.html">New XM investigation: paired control interventions, lift-first feasibility, training noise and captured contacts →</a></p>'
        if (ROOT / "artifacts/xm_diagnosis/index.html").exists()
        else ""
    )
    result_rows = []
    for name in ("legacy_ff_yaw0", "legacy_ff_yaw1"):
        result = read(f"{name}/results.json", {})
        evaluations = result.get("evaluations", [])
        if not evaluations:
            result_rows.append(
                f"<tr><td>{name}</td><td colspan='6'>Pending — no measured validation success rate yet.</td></tr>"
            )
        for evaluation in evaluations:
            count = evaluation["successes"]
            variants = "".join(
                f"<td>{evaluation['per_variant'][variant]['successes']}/128</td>"
                for variant in ("xm", "xp", "ym", "yp")
            )
            result_rows.append(
                f"<tr><td>{name}</td><td>{evaluation['repeat']}</td><td>{count}/512 ({count / 512:.2%})</td>{variants}</tr>"
            )
    control_rows = "".join(
        f"<tr><td>{'Lift first, then traverse' if row['lift_first'] else 'Original approach'}</td>"
        f"<td>{'Success' if row['held_success'] else 'Failure'}</td><td>{row['steps'] * 0.04:.2f} s</td>"
        f"<td>{1000 * row['final_body_error_m']:.3f} mm</td>"
        f"<td>{row['warm_ik_position_residual_m']:.2e} m</td></tr>"
        for row in diagnosis["cases"]
    )
    compare = [
        (
            "Dense reward",
            "−log(1 + 100·distance)",
            "3·change in a two-scale potential",
            "−log(1 + 100·physical distance)",
        ),
        (
            "Success reward",
            "+1000 replaces dense cost",
            "+10, plus potential change",
            "+1000 added to dense cost",
        ),
        (
            "Success sampling",
            "Derived position, instantaneous <2 mm",
            "Current qpos, <2 mm for 3 samples",
            "Current qpos, <2 mm for 3 samples",
        ),
        (
            "Actor / critic",
            "Feedforward 256–256–128",
            "GRU128 + 128–128 MLP",
            "Feedforward 256–256–128",
        ),
        (
            "Initial standard deviation",
            "1.0",
            "1.0 in latest exploration repair; 0.4 originally",
            "1.0",
        ),
        (
            "Initial LR / entropy",
            "0.001 adaptive / 0.005",
            "0.0003 adaptive / 0.01 in latest repair",
            "0.001 adaptive / 0.005",
        ),
        ("Epochs / minibatches", "5 / 4", "4 / 8", "5 / 4"),
        (
            "Training transitions",
            "196,608,000 globally across 4 GPUs",
            "9,228,288 latest repair; 18,432,000 original comparison",
            "9,228,288 per case; two cases",
        ),
        (
            "Episode / inspection",
            "5 s / no enforced pause in saved pl3 config",
            "3.5 s / first 1 s manipulation pause",
            "3.5 s / first 1 s manipulation pause",
        ),
        (
            "Control / physics rate",
            "45.45 Hz / dt 0.0044 s",
            "25 Hz / dt 0.002 s",
            "25 Hz / dt 0.002 s",
        ),
        (
            "Images",
            "96×96, 71° FOV, dual views concatenated into one encoder",
            "128×96, 48.46° FOV, separate view encoders",
            "Current image pipeline retained",
        ),
        (
            "Physical shortcut",
            "Variant-dependent inertials in pre-fix pl3",
            "Equalized inertials",
            "Equalized inertials retained",
        ),
        (
            "Timeout handling",
            "Infinite-horizon bootstrap",
            "Finite-horizon",
            "Current finite-horizon retained",
        ),
        ("Reward scaling by dt", "Disabled", "Disabled", "Disabled"),
        (
            "Manipulator actions",
            "XYZ translation, fixed orientation",
            "XYZ translation, fixed orientation",
            "Case A: XYZ. Case B: XYZ + bounded yaw",
        ),
    ]
    compare_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in compare
    )
    doc = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plug training: diagnosis and repair</title><style>
:root{{color-scheme:light;--ink:#172c37;--muted:#536974;--accent:#076963;--line:#cbd9db}}
*{{box-sizing:border-box}}body{{margin:0;background:#edf3f1;color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1240px;margin:auto;padding:36px 24px 70px}}header{{background:#153c40;color:white;padding:40px;border-radius:20px}}
h1{{font-size:clamp(2rem,5vw,3.4rem);line-height:1.1;margin:12px 0 22px}}h2{{font-size:1.65rem;line-height:1.3}}a{{color:var(--accent)}}header a{{color:#b9f4d8}}
section{{padding:25px 30px;background:white;border:1px solid var(--line);border-radius:14px;margin:22px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}.card{{padding:20px;background:#e7f2ef;border-radius:12px}}
.big{{font-size:2.2rem;font-weight:750;line-height:1.2}}.muted,small{{color:var(--muted)}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:.92rem}}th,td{{padding:12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#eaf2f0}}.compare{{min-width:850px}}
code{{font-family:ui-monospace,monospace;font-size:.9em}}.callout{{border-left:5px solid #c48b22;padding:8px 20px;background:#fff6df}}li{{margin:.65em 0}}nav{{display:flex;flex-wrap:wrap;gap:20px;margin:20px 0}}@media print{{body{{background:white}}section{{break-inside:avoid}}main{{padding:0}}}}
</style><main><header><div>SO101 · OCTOBER 9, 2026 · REPAIR EVIDENCE</div><h1>{lead_title}</h1>
<p>{lead_text}</p>{diagnosis_link}
<a href="../training_results/index.html">← Full training results and hyperparameter search</a></header>
<nav><a href="#findings">Findings</a><a href="#recipe">Reward and training differences</a><a href="#contact">XM contact failure</a><a href="#implementation">Changes</a><a href="#experiment">New experiment</a><a href="#evidence">Evidence</a></nav>
<section id="findings"><h2>What the evidence says</h2><div class="grid">
<div class="card"><div class="big">0 / 128</div>XM successes in each latest learned-policy repeat</div>
<div class="card"><div class="big">119 / 128</div>XM successes with the existing privileged Warp controller</div>
<div class="card"><div class="big">21.3×</div>Legacy global training budget versus the latest 9.23M repair</div></div>
<p>The privileged controller solves 503/512 total randomized starts with the existing three translation actions. Extra rotation is therefore not required to make most sampled XM starts physically solvable. It may help contact recovery and visual access, but it also increases the learning problem.</p>
<div class="callout"><strong>Old high success rates are not a clean target baseline.</strong> The legacy task README marks the 94.0% wrist / 86.2% static / 100% active seed-0 results superseded: variant-dependent mass and inertia produced a sag signal that policies could use. Its active policy still scored 97.4% with camera motion frozen. These are historical documented figures, not newly reproduced measurements. <a href="sources/legacy-insertion-README.md">Read the source</a>.</div>
<p>The proposed 80–90% static and 90–100% active targets remain hypotheses. Neither this repair nor the older results establish that ordering under matched geometry, held-out starts, training budgets and robustness tests.</p></section>
<section id="recipe"><h2>The old reward was materially different</h2><p>For insertion, the legacy source computes <code>−log(1 + 100·distance)</code> and replaces it with <code>+1000</code> when the body is within 2 mm. The previous log-reward trial in this repository used the smaller recurrent architecture and initial std 0.4, so it was not a reproduction of the old training recipe.</p>
<div class="scroll"><table class="compare"><thead><tr><th>Setting</th><th>Saved legacy pl3</th><th>Current / latest repair</th><th>Prepared recipe comparison</th></tr></thead><tbody>{compare_rows}</tbody></table></div>
<p class="muted">The new <code>legacy_ff</code> profile recovers network widths and PPO settings. It deliberately retains current geometry, image encoders, horizon, observation schema and physical hold scoring. Consequently this is a legacy-style experiment, not an exact historical reproduction.</p></section>
<section id="contact"><h2>A hard XM approach failure has a concrete explanation</h2><p>Native MuJoCo, seed 10011, unchanged target and three translation actions. The original controller moves horizontally while lifting; a prong catches the socket. Its commanded pose is reachable to numerical precision, but contact prevents the physical arm from following.</p>
<div class="scroll"><table><thead><tr><th>Controller</th><th>Three-sample result</th><th>Elapsed</th><th>Final body error</th><th>IK position residual</th></tr></thead><tbody>{control_rows}</tbody></table></div>
<p>At the jam, prong-to-socket contacts carry 7.49 N and 6.50 N normal forces. Holding XY while first lifting the TCP to 90 mm removes this failure, then the same approach/insert controller succeeds. This is a one-case privileged diagnostic, not a learned-policy improvement or a new 512-episode success rate. The production scripted controller and training reset were not changed by this diagnostic.</p>
<p>The learned-policy evidence is different: seven XM episodes briefly reach &lt;2 mm but do not hold for three samples; median closest XM error is 2.248 mm. Near-contact alignment and stable insertion remain unresolved. <a href="xm-native-diagnosis.json">Full contact and IK measurements</a>.</p></section>
<section id="implementation"><h2>Implemented and checked</h2><ul>
<li><strong>Optional wrist yaw:</strong> a fourth manipulator command, bounded to ±30° with a 3°/step rate limit. It uses the existing arm joints and respects the one-second inspection pause.</li>
<li><strong>Orientation-aware success:</strong> yaw runs require physical orientation error &lt;2° as well as &lt;2 mm position error for three samples. The evaluator retains all three orientation measurements. Position-only historical checkpoints retain their original schema.</li>
<li><strong>Orientation shaping:</strong> yaw runs add an angle cost with a 30 mm lever-arm scale to the log reward. This means the pair tests a usable yaw mode, including its scoring/shaping, rather than isolating action dimension alone.</li>
<li><strong>Full feedforward recipe:</strong> both actor and critic are feedforward; simply setting the old <code>--memory none</code> left the critic recurrent. Resume validation protects architecture, action dimensions and objective.</li>
<li><strong>Validation:</strong> 166 non-GPU regression tests passed, then 15 orientation/control checks and 71 targeted tests after adding configuration preservation and checkpoint save/load coverage. Native tests rotate and realign the physical plug and insert all four variants. GPU validation status: <strong>{html.escape(smoke.get("status", "not started"))}</strong>; passed artifact: <strong>{bool(gpu.get("passed"))}</strong>.</li></ul></section>
<section id="experiment"><h2>Bounded experiment and decision rule</h2><p><strong>Status: {html.escape(status)}. Slurm job: {html.escape(str(job))}.</strong> Two fresh seed-0 fixed-view-7 runs, one GPU sequentially, each 751 updates × 512 environments × 24 steps = 9,228,288 transitions. Total: 18,456,576. No automatic budget extension or active/wrist sweep.</p>
<p>Both runs use the recovered feedforward/log recipe. The second enables yaw plus orientation scoring/shaping. Each final checkpoint receives two 512-episode physical validations: 128 episodes for each variant, reset seeds 10000–10003. Representative trajectories and videos are captured. The competence gate is at least 103/128 successes for every variant in both repeats.</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Repeat</th><th>Overall</th><th>XM</th><th>XP</th><th>YM</th><th>YP</th></tr></thead><tbody>{"".join(result_rows)}</tbody></table></div>
<p>Use measured per-variant gains and learning trends to decide whether to extend training. Once plug competence is demonstrated, run matched static/active/wrist conditions with equal manipulation DOFs, several training seeds and the same robustness suite. A rising reward alone is insufficient.</p></section>
<section id="evidence"><h2>Reproducible evidence</h2><ul><li><a href="recipe-plan.json">Exact commands, budgets and complete source/asset hashes</a></li>
<li><a href="sources/legacy-rewards.py">Legacy reward source snapshot</a> · <a href="sources/legacy-run-inventory.json">Saved run configurations and checkpoint provenance</a></li>
<li><a href="sources/legacy-insertion-README.md">Legacy task README, including sag diagnosis</a></li>
<li><a href="xm-native-diagnosis.json">Native XM approach/contact experiment</a></li>
<li><a href="preflight.json">CPU checks and frozen-source verification</a> · <a href="gpu-validation.json">Rendered GPU PPO and checkpoint checks</a> · <a href="train-submission.json">Training submission and budget</a></li>
<li><a href="../plug_analysis/randomized_feasibility_gpu_verified.json">Existing 512-episode Warp control verification</a> · <a href="../training_results/index.html">Full results and hyperparameter report</a></li></ul>
<p class="muted">Built {now}. Frozen source is an uncommitted snapshot: file hashes and the saved patch identify it; the base Git revision alone does not. Validation repeats measure execution variability of one training seed, not uncertainty across independently trained models.</p></section></main></html>"""
    (OUT / "index.html").write_text(doc)
    (OUT / "report-provenance.json").write_text(
        json.dumps(
            {
                "built_at": now,
                "sources": provenance,
                "gpu_passed": bool(gpu.get("passed")),
                "training_status": status,
                "job_id": job,
                "planned_transitions": plan["total_training_transitions"],
            },
            indent=2,
        )
        + "\n"
    )
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
