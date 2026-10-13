"""Build transfer/push evidence pages from saved results, without policy inference."""

import argparse
import base64
import html
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join("<th>" + html.escape(h) + "</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + str(v) + "</td>" for v in row) + "</tr>" for row in rows
        )
        + "</tbody></table></div>"
    )


def atomic(path, value):
    temp = path.with_suffix(f".tmp-{os.getpid()}")
    temp.write_text(value)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    base, output = root / "artifacts/transfer_push", root / "artifacts/training_results"
    output.mkdir(parents=True, exist_ok=True)
    plan = read(base / "study-plan.json", {})
    suspension = read(base / "suspension.json", {})
    stamp = datetime.now(timezone.utc).isoformat()

    def link(path, label):
        return f'<a href="{html.escape(os.path.relpath(path, output))}">{html.escape(label)}</a>'

    rows, evaluations, states, curves, interventions, media = [], [], [], [], [], []
    labels = {"wrist": "Wrist only", "wrist_static": "Wrist + static", "active": "Wrist + active"}
    for run in plan.get("runs", []):
        directory = base / "runs" / run["id"]
        state = read(directory / "state.json", {})
        submissions = [read(p) for p in sorted(directory.glob("submission-*.json"))]
        status = state.get("status", "submitted" if submissions else "waiting for task competence")
        transitions = state.get("transitions", 0)
        history = []
        if state.get("run_directory"):
            run_dir = Path(state["run_directory"])
        else:
            dirs = list((Path(plan["logs"]) / run["id"]).glob("*/experiment.json"))
            run_dir = dirs[0].parent if len(dirs) == 1 else None
        if run_dir:
            for path in sorted(run_dir.glob("resume_history/*/iterations.jsonl")) + [
                run_dir / "iterations.jsonl"
            ]:
                if path.exists():
                    text = path.read_text()
                    complete = text[: text.rfind("\n") + 1]
                    history.extend(json.loads(line) for line in complete.splitlines())
            if history:
                history = sorted(
                    {r["iteration"]: r for r in history}.values(), key=lambda r: r["iteration"]
                )
                transitions = max(transitions, history[-1]["transitions"])
                curves.append((run, history))
        recent = {}
        for evaluation in state.get("evaluations", []):
            path = Path(evaluation["path"])
            raw = read(path)
            assert raw["successes"] == sum(raw["outcomes"]) == evaluation["successes"]
            assert raw["episodes"] == len(raw["outcomes"]) == evaluation["episodes"]
            audit = raw["termination_audit"]
            assert (
                audit["success_hold_violations"]
                == audit["success_" + run["task"] + "_violations"]
                == 0
            )
            recent[evaluation["occlusion"]] = evaluation
            ci = raw["wilson_95"]
            evaluations.append(
                [
                    run["task"],
                    labels[run["condition"]],
                    run["seed"],
                    f"{evaluation['transitions']:,}",
                    evaluation["split"],
                    evaluation["occlusion"],
                    f"{raw['successes']}/{raw['episodes']} ({100 * raw['success_rate']:.1f}%)",
                    f"{100 * ci[0]:.1f}–{100 * ci[1]:.1f}%",
                    link(path, "Raw evaluation"),
                ]
            )

        def score(mode, recent=recent):
            r = recent.get(mode)
            return (
                f"{100 * r['success_rate']:.1f}% ({r['successes']}/{r['episodes']})"
                if r
                else "Pending"
            )

        target = (
            math.ceil(plan["min_transitions"] / (plan["num_envs"] * 24)) * plan["num_envs"] * 24
        )
        if status == "completed":
            assert transitions >= plan["min_transitions"]
        rows.append(
            [
                run["task"],
                labels[run["condition"]],
                run["seed"],
                html.escape(status),
                f"{transitions:,} / {target:,}",
                score("clean"),
                score("dynamic"),
                ", ".join(str(r.get("job_id", "submission failed")) for r in submissions) or "—",
            ]
        )
        states.append(
            dict(
                run=run,
                status=status,
                observed_transitions=transitions,
                target_transitions=target,
                evaluations=state.get("evaluations", []),
                submissions=submissions,
                competence_gate=state.get("competence_gate"),
            )
        )
        for name, intervention in state.get("interventions", {}).items():
            interventions.append(
                [
                    run["task"],
                    run["seed"],
                    name,
                    f"{intervention['successes']}/{intervention['episodes']}",
                    link(Path(intervention["path"]), "Raw intervention"),
                ]
            )
        for path in sorted(directory.glob("videos-*/representatives.json")):
            for record in read(path)["records"]:
                movie = Path(record["outside_video"])
                inputs = Path(record["policy_video"])
                title = html.escape(
                    run["id"] + " · " + path.parent.name + " · " + str(record.get("success", ""))
                )
                media.append(
                    f'<h3>{title}</h3><video controls preload="none" width="640" src="{html.escape(os.path.relpath(movie, output))}"></video>'
                    + "<p>"
                    + link(inputs, "Actual policy inputs")
                    + "</p>"
                )
    figure = ""
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for run, history in curves:
            history = history[:: max(1, len(history) // 2000)]
            xs = [r["transitions"] / 1e6 for r in history]
            for axis, key in zip(
                axes, ("train/Train/mean_reward", "train/Episode_Metrics/success_rate"), strict=True
            ):
                ys = [r.get(key, float("nan")) for r in history]
                axis.plot(xs, ys, label=run["id"], linewidth=1)
        axes[0].set_ylabel("Training episode return")
        axes[1].set_ylabel("Native training success metric")
        for axis in axes:
            axis.set_xlabel("Million environment transitions")
            axis.grid(alpha=0.2)
        axes[1].legend(fontsize=7)
        data = io.BytesIO()
        fig.savefig(data, format="svg", bbox_inches="tight")
        fig.savefig(output / "transfer_push_learning.pdf", bbox_inches="tight")
        plt.close(fig)
        figure = (
            '<img alt="Training reward and success curves" src="data:image/svg+xml;base64,'
            + base64.b64encode(data.getvalue()).decode()
            + '">'
        )
    audits = []
    for path in sorted(base.glob("gpu-audit-*/summary.json")):
        audit = read(path)
        for task, result in audit.items():
            experiment = read(path.parent / (task + "-physics.json"))["experiment"]
            audits.append(
                [
                    path.parent.name,
                    task,
                    experiment["reset_xy_jitter"],
                    f"{result['successes']}/{result['episodes']}",
                    str(result["smoke"]["passed"]),
                    link(path, "Raw audit"),
                ]
            )
    completed = sum(s["status"] == "completed" for s in states)
    active = sum(bool(s["submissions"]) and s["status"] != "completed" for s in states)
    summary = dict(
        updated_utc=stamp,
        completed_runs=completed,
        submitted_incomplete_runs=active,
        planned_runs=len(states),
        minimum_transitions_per_run=plan.get("min_transitions", 1_000_000_000),
        runs=states,
        claim="Active perception advantage is not established; comparisons require matched completed runs.",
    )
    atomic(output / "transfer_push_summary.json", json.dumps(summary, indent=2) + "\n")
    fragment = f"""<section id="transfer-push-study"><h2>Transfer and push: billion-step study</h2>
<p><strong>Minimum 1 billion environment transitions per policy run.</strong> {completed}/{len(states)} planned runs completed;
{active} submitted runs remain incomplete. Three training seeds per task/camera configuration are planned.</p>
<p>{link(output / "transfer_push.html", "Open live transfer/push results, progress and evaluations")} ·
{link(output / "transfer_push_summary.json", "Machine-readable status")}</p>
<p>The earlier transfer pilot scored 0/512 clean and 0/512 dynamic after 3.08M transitions. New controls/rewards and physical push scoring are a separate study. An active-camera advantage has not yet been established.</p></section>"""
    if suspension:
        notice = (
            "<p><strong>RGB STUDY STOPPED: privileged state-based reward/learnability checks come first.</strong> "
            "The two RGB checkpoints and logs are preserved. No further RGB runs are queued or automatically enabled. "
            + link(base / "suspension.json", "Stop/checkpoint record")
            + "</p>"
        )
        fragment = fragment.replace("</h2>", "</h2>" + notice, 1)
        fragment = fragment.replace(
            f"{active} submitted runs remain incomplete.",
            f"{active} submitted runs were stopped before completion.",
        )
    atomic(output / "transfer_push_section.html", fragment)
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transfer and push · Training results</title><style>
body{{margin:0;background:#f2f5f7;color:#203040;font:16px/1.55 system-ui}}main{{max-width:1400px;margin:auto;padding:30px}}
header{{padding:28px;background:#12333d;color:white;border-radius:12px}}h1{{font-size:38px;line-height:1.15}}h2{{margin-top:32px}}
a{{color:#167185}}header a{{color:#b7ecf2}}section{{background:white;padding:22px;margin-top:20px;border-radius:10px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}
th{{background:#e7f0f1}}.scroll{{overflow-x:auto}}img{{max-width:100%}}.note{{color:#526777}}code{{overflow-wrap:anywhere}}
</style><main><header><p>SO-101 · physical manipulation · {stamp}</p><h1>Transfer and push</h1>
<p>Wrist only / wrist + static / wrist + active · 1B transitions per run · three planned training seeds</p>
<a href="index.html">All training results</a></header>
{fragment}<section><h2>Policy runs</h2>
<p>Completed: <strong>{completed}</strong>. Submitted and incomplete: <strong>{active}</strong>. Pending scores are not zeros.
Live transitions come from complete optimizer log records; evaluation scores refer to the checkpoint budget shown below.</p>
{table(["Task", "Sensing", "Seed", "Status", "Completed / target transitions", "Latest clean", "Latest dynamic", "Slurm jobs"], rows)}
</section><section><h2>Protocol and task changes</h2>
<p>Each fresh policy uses compact CNN + GRU PPO, 24 steps/update, {plan.get("num_envs", "pending")} parallel environments,
initial Gaussian std 1.0, adaptive learning rate 0.0003, entropy 0.01, four epochs and eight minibatches.
Training uses dynamic optical occlusion, 96×72 RGB, a 12-second horizon, and independently randomized object/goal XY within ±20 mm.
All conditions share manipulation controls, reward, task distribution, recurrent memory, and total transition budget.</p>
<p>Transfer commands XYZ plus gripper; push commands XYZ with a rigid pushing tip. Camera-arm actions remain five joint increments.
Dense stage rewards encourage approach, pickup, transport and release; success adds 100. Transfer requires bilateral grasp/lift history,
floor support, release, settling and withdrawal for five samples. Push requires the actual tip to withdraw more than 25 mm,
no load-bearing manipulator contact, goal tolerances and low speed for three samples. Scoring reads current integrated qpos.</p>
<p>The seed-zero wrist+static policy for each task trains first. At least 80% success on both clean and dynamic 512-episode validation
sets enables that task's other eight runs. Every started policy continues to at least 1B transitions. This gate is a competence check;
it is not evidence of a sensing advantage. Validation reset seeds start at 30000; final 1024-episode test sets start at 80000.</p>
<p>Static transfer view 4 was chosen from the earlier segmentation screen; push uses home view 0. These are specified baselines,
not learned-success-selected best-static cameras. Static viewpoint search, initial-inspection and prescribed-scan alternatives
remain limitations of a broad claim that continuous active motion is necessary.</p>
<p>Slurm time-limit signals save a checkpoint and schedule a continuation with the same model, optimizer and W&amp;B identity.
Simulator/RNG/episode/GRU state resets on resume. Each continuation is logged; total budgets count environment transitions,
not simulator substeps or vector steps.</p>
{link(base / "study-plan.json", "Immutable study plan") if plan else "Training plan is being prepared."}
</section><section><h2>Held-out evaluations</h2>
<p>Wilson intervals describe sampled episodes for one trained policy; they do not measure variation between training seeds.</p>
{table(["Task", "Sensing", "Seed", "Transitions", "Split", "Occlusion", "Success", "95% Wilson", "Evidence"], evaluations) if evaluations else "<p>No new held-out evaluation has completed.</p>"}
</section><section><h2>Learning curves</h2>{figure or "<p>Training telemetry pending.</p>"}
<p class="note">Training return and reset-batch success are diagnostics. Use held-out physical success to compare policies.</p>
</section><section><h2>Feasibility and pipeline validation</h2>
{table(["Audit snapshot", "Task", "XY jitter (m)", "Scripted successes", "RGB PPO + reload passed", "Evidence"], audits)}
<p>These are privileged scripted controllers and two-update PPO smoke checks. They do not establish learned task competence.
The first GPU audit used ±8 mm and TCP-based push withdrawal; the later audit uses ±20 mm and actual-tip withdrawal.</p>
</section><section><h2>Active-camera interventions</h2>
<p>Final active checkpoints are evaluated with the camera frozen after one second and with the external image held after one second.
These diagnose camera/image dependence of the trained policy; they are not retrained baselines.</p>
{table(["Task", "Seed", "Intervention", "Success", "Evidence"], interventions) if interventions else "<p>Pending final active checkpoints.</p>"}
</section><section><h2>Representative captured rollouts</h2>{"".join(media) or "<p>Pending final evaluations.</p>"}
</section><section><h2>Preserved failed transfer baseline</h2>
<p>The previous joint-control / potential-difference pilot completed 501 updates (3,078,144 transitions), with 0/512 successes in
both clean and dynamic validation. All 1024 episodes failed before lifting. It remains separate from this changed task/control/reward recipe.</p>
{link(root / "artifacts/transfer_repair/pilot-runtime.json", "Completed pilot evidence")}
</section><p class="note">Results are generated from local saved evidence. No supported active-camera ranking yet.</p></main></html>"""
    if suspension:
        document = document.replace(
            "Every started policy continues to at least 1B transitions.",
            "This was the earlier launch protocol; it is now suspended.",
        )
        document = document.replace(
            "sets enables that task's other eight runs.",
            "sets was intended to enable expansion; that automatic expansion is now disabled.",
        )
    atomic(output / "transfer_push.html", document)
    # Keep the established report destination useful between full report rebuilds.
    main_path = output / "index.html"
    if main_path.exists():
        text = main_path.read_text()
        start, end = "<!-- transfer-push-live-start -->", "<!-- transfer-push-live-end -->"
        if start in text:
            text = text[: text.index(start)] + text[text.index(end) + len(end) :]
        content = start + fragment + end
        text = (
            text.replace("<main>", "<main>" + content, 1)
            if "<main>" in text
            else text.replace("</header>", "</header>" + content, 1)
        )
        atomic(main_path, text)
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
