#!/usr/bin/env python3
"""Audit saved XM interventions and build an HTML report; never run a learner."""

import base64
import hashlib
import html
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/xm_diagnosis"
VARIANTS = ("xm", "xp", "ym", "yp")
POLICIES = {
    "progress_gru": "Progress reward · GRU · XYZ",
    "legacy_ff_yaw0": "Legacy reward · feedforward · XYZ",
    "legacy_ff_yaw1": "Legacy reward · feedforward · XYZ + yaw",
}
MODES = {
    "baseline": "Saved deterministic policy",
    "slew3mm": "Policy with 3 mm target slew limit",
    "hold_at_2s": "Hold measured TCP at 2 seconds",
    "oracle_hold_at_2s": "True final target from 2 seconds (privileged)",
    "oracle_path": "True target; raise while traversing (privileged)",
}
APPROACH = {
    "oracle_lift_first": "Lift first, then align and insert",
    "oracle_lift_first_noise": "Same controller + saved training noise",
    "oracle_lift_first_noise10": "Same controller + 10% training noise",
    "oracle_lift_first_delta5mm_noise": "5 mm increments + saved normalized noise",
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image(path, alt):
    kind = "svg+xml" if path.suffix == ".svg" else "png"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/{kind};base64,{data}">'


def table(headers, rows, attributes=""):
    head = "".join(f"<th>{h}</th>" for h in headers)
    return f'<div class="scroll"><table {attributes}><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def row(cells, attributes=""):
    return f"<tr {attributes}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def count_cells(data):
    summary = data["summary"]
    wins = sum(s["successes"] for s in summary.values())
    total = len(data["records"])
    return [f"{wins}/{total} ({wins / total:.1%})"] + [
        f"{summary[v]['successes']}/{summary[v]['episodes']}" for v in VARIANTS
    ]


def audit():
    controls, approaches, evidence = {}, {}, []
    manifests = read(OUT / "checkpoints.json")
    hashes = {name: digest(Path(cp)) for name, cp in manifests.items()}
    paired = {}
    for folder, destination in (("controls", controls), ("approach", approaches)):
        for path in sorted((OUT / folder).glob("*.json")):
            data = read(path)
            policy = path.name.split("-", 1)[0]
            assert data["checkpoint_sha256"] == hashes[policy], path
            assert data["learner_updates"] == 0
            assert data["experiment"]["success_state_sample"] == "current_qpos"
            assert data["experiment"]["episode_seconds"] == 3.5
            keys = [(r["seed"], r["world"]) for r in data["records"]]
            assert len(set(keys)) == len(keys) == 128 * len(data["seeds"])
            for index, seed in enumerate(data["seeds"]):
                key = policy, seed
                initial = data["initial_hashes"][index * 128 : (index + 1) * 128]
                assert len(initial) == 128 and paired.setdefault(key, initial) == initial
            for variant in VARIANTS:
                records = [r for r in data["records"] if r["variant"] == variant]
                summary = data["summary"][variant]
                assert len(records) == summary["episodes"] == 32 * len(data["seeds"])
                assert sum(r["success"] for r in records) == summary["successes"]
                for r in records:
                    assert r["variant"] == VARIANTS[r["world"] % 4]
                    assert r["success"] or r["failure"] or r["timeout"]
                    assert np.isfinite(r["body"]).all() and np.isfinite(r["goal"]).all()
                    assert np.isclose(
                        np.linalg.norm(np.subtract(r["body"], r["goal"])),
                        r["qpos_position_error_m"],
                        atol=1e-7,
                    )
                    if r["success"]:
                        assert r["success_hold_steps"] >= 3
                        assert all(
                            math.isfinite(e) and e < 0.002 for e in r["last_three_manager_errors_m"]
                        )
                        if data["experiment"]["plug_wrist_yaw"]:
                            assert all(
                                a < math.radians(2) for a in r["last_three_orientation_errors_rad"]
                            )
            destination[policy, data["mode"]] = data
            evidence.append(
                {"file": str(path.relative_to(OUT)), "sha256": digest(path), "episodes": len(keys)}
            )
    assert len(controls) == 15 and len(approaches) == 4
    sources = []
    for name in (
        "submission.json",
        "continuation-submission.json",
        "approach-submission.json",
        "delta-submission.json",
    ):
        submission = read(OUT / name)
        source = Path(submission["source"])
        for relative, expected in submission["source_sha256"].items():
            assert digest(source / relative) == expected, (name, relative)
        sources.append(
            {
                "manifest": name,
                "job_id": submission["job_id"],
                "verified_files": len(submission["source_sha256"]),
            }
        )
    noise = {}
    for name, cp in manifests.items():
        saved = torch.load(cp, map_location="cpu", weights_only=True)
        std = saved["actor_state_dict"]["distribution.std_param"].numpy()
        noise[name] = {
            "checkpoint": cp,
            "sha256": hashes[name],
            "normalized_std": std.tolist(),
            "xyz_sigma_mm": (std[:3] * [70, 60.5, 17.5]).tolist(),
            "yaw_sigma_deg": float(std[3] * 30) if len(std) > 3 else None,
        }
    result = {
        "built_at": datetime.now(UTC).isoformat(),
        "cases": evidence,
        "source_snapshots": sources,
        "noise": noise,
        "paired_initial_state_and_image_hashes_match": True,
        "physical_success_violations": 0,
        "learner_updates": 0,
    }
    (OUT / "report-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return controls, approaches, result


def figures(controls):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    for ax, policy in zip(axes, POLICIES, strict=True):
        points = [t for t in controls[policy, "baseline"]["trace"] if t["world"] == 0]
        time = [(t["step"] + 1) * 0.04 for t in points]
        goal_x = points[0]["goal"][0]
        ax.plot(
            time,
            [(t["target"][0] - goal_x) * 1000 for t in points],
            label="Requested TCP X − correct TCP X",
            color="#c86823",
            alpha=0.8,
        )
        ax.plot(
            time,
            [(t["physical_tcp"][0] - goal_x) * 1000 for t in points],
            label="Measured TCP X − correct TCP X",
            color="#167b78",
        )
        ax.axhline(0, color="#455467", linewidth=0.8)
        ax.axvspan(0, 1, color="#eceef2", label="Inspection pause")
        ax.set(ylabel="X error (mm)", title=POLICIES[policy], xlim=(0, 3.52))
        ax.grid(alpha=0.18)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time after reset (seconds)")
    for suffix in ("svg", "pdf"):
        fig.savefig(OUT / f"xm-control-traces.{suffix}")
    plt.close(fig)


def variant_diagram():
    parts = []
    for index, (name, dx) in enumerate((("XM", -45), ("XP", 45))):
        x = 25 + index * 330
        parts.append(
            f'<g transform="translate({x},0)"><text x="125" y="24" text-anchor="middle" font-weight="700">{name}: prong pair {"−" if dx < 0 else "+"}15 mm in X</text><rect x="29" y="44" width="192" height="126" rx="8" fill="#e6edf0" stroke="#64748b"/><path d="M125 36V181" stroke="#64748b" stroke-dasharray="4 4"/><circle cx="125" cy="107" r="3" fill="#233e49"/><rect x="{125 + dx - 33 - 12}" y="95" width="24" height="24" fill="#c09132"/><rect x="{125 + dx + 33 - 12}" y="95" width="24" height="24" fill="#c09132"/><path d="M125 146H{125 + dx}" stroke="#9b5b05" stroke-width="3"/><text x="125" y="204" text-anchor="middle">Housing shifts {"+" if dx < 0 else "−"}15 mm to align</text></g>'
        )
    return (
        '<svg viewBox="0 0 660 225" role="img" aria-label="Underside schematic of XM and XP prong offsets">'
        + "".join(parts)
        + "</svg>"
    )


def main():
    controls, approaches, audit_result = audit()
    figures(controls)
    control_rows, baseline_rows, noise_rows, approach_rows = [], [], [], []
    for policy, title in POLICIES.items():
        data = controls[policy, "baseline"]
        xm = data["summary"]["xm"]
        baseline_rows.append(
            row(
                [
                    title,
                    *count_cells(data),
                    f"{xm['median_closest_mm']:.2f} mm",
                    f"{xm['reached_2mm']}/32",
                    f"{xm['mean_target_step_mm']:.1f} mm",
                ]
            )
        )
        for mode, label in MODES.items():
            data = controls[policy, mode]
            link = f'<a href="controls/{policy}-{mode}.json">{label}</a>'
            control_rows.append(row([title, link, *count_cells(data)], f'data-policy="{policy}"'))
        n = audit_result["noise"][policy]
        noise_rows.append(
            row(
                [
                    title,
                    *[f"{v:.2f} mm" for v in n["xyz_sigma_mm"]],
                    "—" if n["yaw_sigma_deg"] is None else f"{n['yaw_sigma_deg']:.2f}°",
                ]
            )
        )
    for mode, label in APPROACH.items():
        data = approaches["progress_gru", mode]
        approach_rows.append(
            row([f'<a href="approach/progress_gru-{mode}.json">{label}</a>', *count_cells(data)])
        )
    counts = ["Overall", "XM", "XP", "YM", "YP"]
    baseline = table(
        ["Checkpoint", *counts, "XM median closest", "XM ever <2 mm", "XM mean target jump"],
        baseline_rows,
    )
    control_table = table(
        ["Checkpoint", "Intervention", *counts], control_rows, 'id="controls-table"'
    )
    approach_table = table(["Privileged controller", *counts], approach_rows)
    noise_table = table(["Checkpoint", "X σ", "Y σ", "Z σ", "Yaw σ"], noise_rows)
    options = '<option value="all">All checkpoints</option>' + "".join(
        f'<option value="{p}">{t}</option>' for p, t in POLICIES.items()
    )
    galleries = "".join(
        f"<details><summary>{title}: captured XM failure</summary>{image(OUT / 'geometry' / f'{policy}-xm-rgb.png', title + ' actual captured policy observations')}</details>"
        for policy, title in POLICIES.items()
    )
    provenance = "".join(
        f'<li><a href="{s["manifest"]}">Job {s["job_id"]}: frozen source manifest</a> ({s["verified_files"]} files verified)</li>'
        for s in audit_result["source_snapshots"]
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Why XM fails · SO101 diagnosis</title><style>
:root{{color-scheme:light;--ink:#173743;--muted:#556973;--line:#cddbdd;--green:#176d67}}*{{box-sizing:border-box}}body{{margin:0;background:#edf2f2;color:var(--ink);font:16px/1.6 system-ui,sans-serif}}main{{max-width:1260px;margin:auto;padding:30px 22px 70px}}header{{background:#143a43;color:white;padding:35px;border-radius:16px}}h1{{font-size:clamp(2rem,5vw,3.6rem);line-height:1.1}}h2{{line-height:1.3}}a{{color:var(--green)}}header a{{color:#b5eee0}}section{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px 30px;margin:22px 0}}nav{{display:flex;flex-wrap:wrap;gap:20px;margin:22px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px}}.card{{background:#eaf4ef;border-radius:10px;padding:20px}}.big{{font-size:2.2rem;font-weight:750}}.note{{background:#fff3d9;border-left:4px solid #b77a15;padding:12px 18px}}.muted,small{{color:var(--muted)}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{padding:11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#eaf1f1}}img,svg{{max-width:100%;height:auto}}select{{font:inherit;padding:8px;max-width:100%}}summary{{cursor:pointer;font-weight:650;padding:15px 0}}li{{margin:10px 0}}code{{font-size:.9em}}@media print{{body{{background:white}}main{{padding:0}}section{{break-inside:avoid}}nav,select{{display:none}}}}
</style></head><body><main><header><div>SO101 · PLUG INSERTION · OCTOBER 9, 2026</div><h1>XM is solvable.<br>Approach and precision control are failing.</h1><p>The saved policies still score zero XM successes. Controlled GPU interventions identify socket jamming and unstable commands, and expose training noise far larger than the insertion tolerance. A lift-first controller succeeds on every tested start using the existing three translation actions.</p><a href="../training_results/index.html">Training results and hyperparameter search</a> · <a href="../plug_repair/index.html">Legacy recipe and yaw experiment</a></header>
<nav><a href="#variants">XM vs XP</a><a href="#learned">Learned failures</a><a href="#interventions">Interventions</a><a href="#noise">Exploration scale</a><a href="#geometry">Contacts and images</a><a href="#next">What to change</a><a href="#evidence">Audit</a></nav>
<section id="variants"><h2>XM and XP describe prong offsets</h2><p><b>XM = X minus; XP = X plus.</b> The prong pair is shifted −15 mm or +15 mm in the plug’s local X direction, relative to an identical housing. YM and YP are the corresponding Y offsets. In the socket-aligned orientation, the correct housing position is <code>socket center − prong offset</code>: XM needs +15 mm in X; XP needs −15 mm. The diagram shows the underside; both prongs are 8 mm wide and 22 mm apart.</p>{variant_diagram()}<p>The camera image’s left/right direction depends on the view. These labels are geometry variants, not camera modes, training versions or motion commands. Mass and inertia are equalized across variants; the prongs carry the visual identity cue.</p></section>
<section id="learned"><h2>The learned controllers have not been repaired</h2><p>Fresh deterministic audit: 128 episodes per checkpoint, seed 10000, 32 starts per variant. Same wrist+static view7. These smaller diagnostic samples are separate from the published two 512-episode validations: the new feedforward XYZ run scored 77/512 (all YP), while XYZ+yaw scored 0/512. Both repeats agreed; every XM repeat was 0/128.</p>{baseline}<p>“Closest” is the best physical body position reached in an episode, not a success. Success requires distance &lt;2 mm for three consecutive samples. The yaw-enabled run additionally requires orientation error &lt;2° throughout those samples. Its reward also includes orientation: this is not a pure action-DOF comparison.</p><p>The GRU XM median gets within 2.25 mm but only 1/32 episodes crosses 2 mm even once; none holds. The feedforward XYZ policy usually remains much farther away. The yaw policy gets close more often but fails to settle. The reported target jumps average the late-episode samples (t ≥1.5 seconds), each 40 ms apart; they are requested TCP motion, not physical object displacement.</p>{image(OUT / "xm-control-traces.svg", "Requested and measured TCP X in one representative XM audit episode per policy")}<p class="muted">One representative world per checkpoint, seed 10000; these plots are examples, not population averages. Correct TCP X equals correct housing X because the grip offset has zero X component. <a href="xm-control-traces.pdf">Download plot PDF</a>.</p></section>
<section id="interventions"><h2>Interventions separate a bad approach from a bad final command</h2><p>Each row below uses the same checkpoint and identical initial physics/image hashes within that checkpoint. Policy weights remain frozen. “True target” interventions are privileged diagnostics that bypass learned target estimation.</p><label for="policy">Show checkpoint </label><select id="policy">{options}</select>{control_table}<ul><li><b>Slowing the existing policy is not a repair:</b> a 3 mm slew limit reduces performance. Holding its measured TCP at 2 seconds also mostly fails. Changed action dynamics require training under those dynamics.</li><li><b>Correcting the final command helps some trajectories:</b> at 2 seconds, a true-target command rescues 18/32 GRU XM episodes and 25/32 yaw episodes, but only 3/32 feedforward XYZ episodes. The latter has usually already reached a state that this simple intervention cannot recover.</li><li><b>Approach direction matters:</b> raising while moving sideways solves 29/32 XM and 32/32 of every other variant. The remaining XM starts require housing X around +58 to +64 mm and stall about 63 mm from the target. Lifting in place before traversing removes these failures in the tested set.</li></ul>
<h3>Four-seed confirmation and noise stress test</h3><p>512 episodes per setting, 128 per variant, seeds 10000–10003. All use XYZ only, the original 3.5-second horizon (including the 1-second inspection pause), normal physical termination/reset, and the original 2 mm / three-sample criterion.</p>{approach_table}<div class="note"><b>512/512 is scripted feasibility, not a trained success rate.</b> The controller receives the true target and uses robot state to lift, align and descend. It proves that extra DOFs and a longer episode are not necessary for these sampled starts. It does not establish that the visual policy can infer the target or learn this strategy.</div></section>
<section id="noise"><h2>Training exploration is much larger than the tolerance</h2><p>Normalized absolute XYZ actions span 140 ×121 ×35 mm. Thus one action unit corresponds to 70, 60.5 and 17.5 mm. Saved Gaussian standard deviations translate to these physical command scales before clipping and rate limits:</p>{noise_table}<p>The best GRU checkpoint explores with approximately 17.1 mm X and 13.5 mm Y standard deviation against a 2 mm success threshold. The feedforward XYZ model has almost collapsed Y exploration (0.19 mm) while retaining 27.6 mm X noise. The yaw run retains 22.6° nominal yaw noise against its 2° orientation threshold; yaw motion is separately limited to 3° per step.</p><p>Injecting the GRU’s saved noise into the otherwise perfect controller gives <b>0/512</b>. Reducing that noise tenfold gives <b>447/512 (87.3%)</b>; applying the same normalized noise to bounded 5 mm increments gives <b>449/512 (87.7%)</b>. Both alternatives retain failures. This is direct evidence that the exploration/control scale can prevent precise insertion, even with perfect target information.</p><div class="note"><b>Evaluation is already deterministic.</b> The failing baseline uses the actor mean. Lowering the saved standard deviation at evaluation would not fix those means. The noise experiment diagnoses the learning setup; it does not prove how much of the learned failure it caused. The observed deterministic command reversals remain a separate failure.</div><p>All noise settings start from matched reset states and images. Later random streams can diverge because successful worlds reset at different times; noise realizations are not claimed to be identical across settings. These are four reset seeds of one checkpoint’s noise parameters, not independent training seeds.</p></section>
<section id="geometry"><h2>Failures include real prong–socket edge contacts</h2><p>Captured MuJoCo-Warp poses were loaded into native MuJoCo for geometry inspection without resimulating the trajectory. Contact pairs here are reconstructed geometry at those poses, not measured GPU contact forces.</p><ul><li><b>Feedforward XYZ example:</b> the closest captured frame is 2.329 mm from the target; the plug is tilted about 2° in both roll and pitch. Prongs overlap the front aperture edge by about 0.26 mm. The final captured position is 21.3 mm away.</li><li><b>Feedforward yaw example:</b> at 1.80 seconds, position error reaches 1.541 mm, but orientation error is about 3.5°, so it correctly fails the yaw criterion. Later commands do not preserve alignment; final captured error is 3.25 mm with edge contacts.</li><li><b>GRU example:</b> one difficult XM capture ends with about 11.8° pitch and contacts against the fixture. Its best captured error is 12.1 mm, so this particular example is worse than the aggregate XM median.</li></ul>{galleries}<p><a href="geometry/geometry.json">All reconstructed contact, aperture-margin and pose records</a>. Captured frames precede their following physics step, so the last image need not equal the final pre-reset audit state.</p></section>
<section id="next"><h2>What the evidence supports changing next</h2><ol><li><b>Make the approach learnable:</b> train lift → align → insert, using a staged reward or a training curriculum. Dense distance reward alone favors moving toward the final point and does not teach collision clearance. Keep any true target or stage labels in reward/critic/training demonstrations, not the deployed visual actor.</li><li><b>Train precision control at a suitable scale:</b> evaluate incremental TCP commands or coarse/fine control, with exploration reduced during fine insertion. The two successful noise interventions motivate this change; neither has yet been validated as a learned policy.</li><li><b>Keep scoring comparable:</b> retain the physical 2 mm hold check and report orientation separately for a fair yaw ablation, or apply the same orientation criterion to both models. Do not silently relax the benchmark.</li><li><b>Then assess learning duration:</b> 9.23M transitions is much smaller than the legacy 196.6M global budget, but a longer run with destabilizing exploration is not an evidence-based first repair. Use a bounded learning experiment, per-variant curves, multiple training seeds, then extend if competence is improving.</li></ol><p>Perception and memory may still limit variant identification; these experiments deliberately bypassed that question. Earlier frozen-feature probes had millimeter-scale localization errors, but that is not proof that the images lack sufficient information. Active-camera superiority and the proposed 80–100% learned targets remain untested hypotheses.</p><p>Historical high scores are also not directly comparable: the legacy README marks them superseded because variant-dependent inertia leaked identity through sag. <a href="../plug_repair/sources/legacy-insertion-README.md">Original diagnosis</a>. No additional learned training or production control/reward change was made during this diagnostic audit.</p></section>
<section id="evidence"><h2>Evidence and reproducibility</h2><p>The report builder rechecks all 19 result files against raw per-episode flags, checkpoint hashes, variant balance, current-qpos distances, three-sample holds and enabled orientation criteria. It also verifies initial state/image pairing and every pinned source file. No success violations were found across 3,968 diagnostic episodes.</p><ul><li><a href="report-audit.json">Verification, evidence hashes and saved noise parameters</a></li><li><a href="checkpoints.json">Exact checkpoint paths</a></li>{provenance}</ul><p>Job 51654119 stopped after four completed cases because the diagnostic helper passed a Python boolean to <code>torch.where</code>. The helper was corrected; job 51655476 preserved and verified those cases, then finished the remaining cases. Jobs 51656582 and 51656972 completed the four 512-episode approach/noise settings. No training jobs were run for this investigation.</p><p class="muted">Built {audit_result["built_at"]}. Rebuild with <code>.venv/bin/python scripts/build_xm_diagnosis_report.py</code>. Geometry images are embedded; source/result links are relative. Conclusions concern this simulator, these checkpoints and the sampled clean starts; no robustness or camera ranking is inferred.</p></section></main><script>document.getElementById('policy').addEventListener('change',e=>{{document.querySelectorAll('#controls-table tbody tr').forEach(r=>r.hidden=e.target.value!=='all'&&r.dataset.policy!==e.target.value)}});</script></body></html>"""
    (OUT / "index.html").write_text(document)
    print(
        json.dumps(
            {
                "report": str(OUT / "index.html"),
                "verified_cases": len(audit_result["cases"]),
                "episodes": sum(r["episodes"] for r in audit_result["cases"]),
                "physical_success_violations": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
