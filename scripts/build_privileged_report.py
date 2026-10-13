"""Report state-based learnability separately from halted RGB experiments."""

import argparse
import base64
import html
import io
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build_manipulation_report import atomic, read, table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    base = root / "artifacts/transfer_push/privileged"
    output = root / "artifacts/training_results"
    output.mkdir(parents=True, exist_ok=True)
    plan = read(base / "plan.json", {})
    suspension = read(root / "artifacts/transfer_push/suspension.json", {})
    review = read(base / "review.json", {})
    reward_audit = read(base / "reward-audit.json", {})
    extension = read(base / "extension-decision.json", {})
    states, evaluations, current, histories, videos, diagnoses = {}, [], [], {}, [], []

    def link(path, label):
        return f'<a href="{html.escape(os.path.relpath(path, output))}">{html.escape(label)}</a>'

    for task in ("transfer", "push"):
        state = read(base / task / "runtime.json", {})
        states[task] = state
        dirs = list((Path(plan.get("logs", base / "logs")) / task).glob("*/experiment.json"))
        history = []
        if dirs:
            directory = dirs[0].parent
            paths = sorted(directory.glob("resume_history/*/iterations.jsonl")) + [
                directory / "iterations.jsonl"
            ]
            for path in paths:
                if path.exists():
                    text = path.read_text()
                    history += [
                        json.loads(line) for line in text[: text.rfind("\n") + 1].splitlines()
                    ]
            histories[task] = sorted(
                {r["iteration"]: r for r in history}.values(), key=lambda r: r["iteration"]
            )
            history = histories[task]
        count = history[-1]["transitions"] if history else state.get("transitions", 0)
        recent = state.get("evaluations", [])[-2:]
        scores = " / ".join(f"{r['successes']}/{r['episodes']}" for r in recent) or "Pending"
        if len(recent) == 2 and recent[0]["transitions"] == recent[1]["transitions"]:
            budget = recent[-1]["transitions"]
            if all(r["success_rate"] >= 0.9 for r in recent):
                verdict = "Meets the provisional 90% competence criterion on both sets. Review trajectories before RGB training."
            elif all(r["successes"] == 0 for r in recent):
                verdict = "No held-out completions. Reward growth alone does not establish task learnability."
            else:
                verdict = "Learned completions are present, but the 90% criterion has not been met on both sets."
            diagnoses.append(f"{task.capitalize()} at {budget:,} transitions: {scores}. {verdict}")
        current.append(
            [
                task,
                html.escape(state.get("status", "prepared")),
                f"{count:,}",
                scores,
                str(state.get("competence_demonstrated", "Pending")),
                state.get("job_id", "—"),
            ]
        )
        for r in state.get("evaluations", []):
            raw = read(Path(r["path"]))
            assert raw["successes"] == sum(raw["outcomes"]) == r["successes"]
            assert raw["episodes"] == len(raw["outcomes"]) == r["episodes"]
            assert raw["termination_audit"]["success_hold_violations"] == 0
            assert raw["termination_audit"]["success_" + task + "_violations"] == 0
            errors = [row["qpos_position_error_m"] for row in raw["termination_audit"]["records"]]
            evaluations.append(
                [
                    task,
                    f"{r['transitions']:,}",
                    r["seed"],
                    f"{r['successes']}/{r['episodes']} ({r['success_rate'] * 100:.1f}%)",
                    f"{statistics.median(errors) * 1000:.1f}",
                    html.escape(json.dumps(r["stages"])),
                    link(Path(r["path"]), "Raw evaluation"),
                ]
            )
        for manifest in sorted((base / task).glob("videos*/representatives.json")):
            updates = (
                int(manifest.parent.name.split("-")[-1]) if "-" in manifest.parent.name else 204
            )
            for record in read(manifest, {}).get("records", []):
                path = Path(record["outside_video"])
                if path.exists():
                    label = "success" if record["expected_success"] else "failure"
                    videos.append(
                        f'<h3>{task}: {label} at {updates * plan["num_envs"] * 24:,} transitions</h3><video controls preload="none" width="640" src="{html.escape(os.path.relpath(path, output))}"></video>'
                    )
    plot = ""
    if histories:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        keys = (
            "train/Train/mean_reward",
            "train/Episode_Metrics/success_rate",
            "train/Episode_Metrics/goal_distance",
            "train/Episode_Metrics/lifted",
        )
        for task, history in histories.items():
            for axis, key in zip(axes.flat, keys, strict=True):
                axis.plot(
                    [r["transitions"] / 1e6 for r in history],
                    [r.get(key, float("nan")) for r in history],
                    label=task,
                )
                axis.set_ylabel(key.split("/")[-1])
                axis.set_xlabel("Million transitions")
                axis.grid(alpha=0.2)
        axes[0, 0].legend()
        b = io.BytesIO()
        fig.savefig(b, format="svg", bbox_inches="tight")
        fig.savefig(output / "privileged_learning.pdf", bbox_inches="tight")
        plt.close(fig)
        plot = (
            '<img alt="State-policy reward, success, goal distance and transfer pickup history" src="data:image/svg+xml;base64,'
            + base64.b64encode(b.getvalue()).decode()
            + '">'
        )
    stamp = datetime.now(timezone.utc).isoformat()
    summary = dict(
        updated_utc=stamp,
        stage="privileged_state_diagnostic",
        rgb_status="stopped",
        tasks=states,
        review=review,
        latest_interpretation=diagnoses,
    )
    atomic(output / "privileged_summary.json", json.dumps(summary, indent=2) + "\n")
    fragment = f"""<section id="privileged-diagnostic"><h2>State-based reward and learnability checks</h2>
<p><strong>The large RGB study is stopped.</strong> New policies receive true object/goal/robot state and contact/stage information,
with the same physical task, action controls, reset distribution, rewards and success criteria.</p>
{table(["Task", "Status", "Completed transitions", "Latest validation scores", "Competence ≥90% twice", "Job"], current)}
<p>{link(output / "privileged.html", "Full state-based results and learning curves")} · {link(base / "plan.json", "Diagnostic plan")}</p></section>"""
    atomic(output / "privileged_section.html", fragment)
    stopped_rows = [
        [
            r["run"],
            f"{r.get('logged_transitions_before_final_stop', r['transitions']):,}",
            f"{r['transitions']:,}",
        ]
        for r in suspension.get("runs", [])
    ]
    review_html = (
        "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in review.get("findings", []))
        or "<p>Awaiting checkpoint evaluations. Reward alone cannot establish task competence.</p>"
    )
    review_html = (
        "".join(f"<p><strong>{html.escape(d)}</strong></p>" for d in diagnoses) + review_html
    )
    if extension:
        review_html += f"<p>{html.escape(extension['reason'])} {link(base / 'extension-plan.json', 'Extension plan')}</p>"
    reward_rows = []
    for task, audit in reward_audit.get("tasks", {}).items():
        cases = audit["cases"]
        assert len({case["initial_physics_sha256"] for case in cases}) == 1
        for case in cases:
            assert case["success_hold_violations"] == case["success_" + task + "_violations"] == 0
            reward_rows.append(
                [
                    task,
                    case["case"],
                    f"{case['terminal_flags']['success']}/{len(case['records'])}",
                    f"{case['mean_return']:.2f}",
                    f"{case['mean_discounted_return']:.2f}",
                ]
            )
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Privileged learnability · Transfer and push</title><style>
body{{font:16px/1.5 system-ui;background:#f2f5f7;color:#203040;margin:0}}main{{max-width:1250px;margin:auto;padding:28px}}
section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}table{{width:100%;border-collapse:collapse;font-size:14px}}
td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#e7f0f1}}.scroll{{overflow-x:auto}}img{{max-width:100%}}
a{{color:#167185}}video{{max-width:100%}}</style><main><h1>Does the task learn with full state?</h1><p>Updated {stamp} · <a href="index.html">All results</a></p>
{fragment}<section><h2>Controlled diagnostic</h2><p>Both actor and critic are feedforward 256×256 MLPs, using normalized privileged state.
There are no RGB encoders or camera renders. PPO uses std1, entropy0.01, adaptive LR0.0003, four epochs/eight minibatches,
2,048 worlds and24 steps/update. Actions, rewards, geometry, 12s horizon, dynamic optical panel and ±20mm reset jitter match the halted RGB runs.</p>
<p>Each task starts from scratch at training seed0. Evaluations near2M,5M and10M transitions use512 episodes at reset seeds30000 and31000.
The initial diagnostic is capped at10,027,008 transitions per task. Any reviewed extension is recorded below; no automatic RGB submission follows.
At least90% success on both final validation sets is the provisional competence criterion.</p>
<p>This changes the actor architecture as well as its information. It checks reward/control/task learnability; it is not a matched RGB-versus-state benchmark.
Failure with privileged state would motivate reward, action-space or optimization diagnosis. Success would justify proceeding to visual learning.</p>
<p>Transfer must physically grasp and lift, then support/release/settle and withdraw for five samples. Push must meet goal tolerances,
withdraw the actual pushing tip, lose manipulator contact and settle for three samples. Reward increase alone does not pass the gate.</p></section>
<section><h2>Interpretation and next action</h2>{review_html}</section>
<section><h2>Held-out physical outcomes</h2>{table(["Task", "Transitions", "Reset seed", "Success", "Median goal error (mm)", "Failure stages", "Evidence"], evaluations) if evaluations else "<p>Pending.</p>"}</section>
<section><h2>Training diagnostics</h2>{plot or "<p>Pending.</p>"}<p>Training metrics use terminal episode state; held-out scores above provide episode-weighted completion.</p></section>
<section><h2>Reward ranking in physical rollouts</h2>
<p>Compare scripted completion with stalled controllers on identical initial physics states (64 resets per case).
Both total return and return discounted at PPO's γ=0.99 are shown. This probes obvious reward traps; it does not establish that PPO can discover the successful strategy.</p>
{table(["Task", "Controller", "Physical success", "Mean return", "Mean discounted return"], reward_rows) if reward_rows else "<p>Pending.</p>"}
{link(base / "reward-audit.json", "Raw reward audit") if reward_rows else ""}</section>
<section><h2>Captured policy rollouts</h2>{"".join(videos) or "<p>Pending the final diagnostic checkpoints.</p>"}<p>These are actual evaluated physics trajectories; state vectors and actions are stored in the capture arrays.</p></section>
<section><h2>Stopped RGB runs</h2>{table(["Run", "Logged transitions", "Preserved checkpoint transitions"], stopped_rows)}
<p>The legacy cancellation handler submitted unwanted continuations. Those jobs were also stopped; the larger logged counts include that extra work.
The earlier checkpoints, optimizer state and all logs are preserved. The launcher now checks the suspension at submission and startup,
and cancellation cannot schedule a continuation. These incomplete runs are not the promised 1B-step camera comparisons.</p>
{link(root / "artifacts/transfer_push/suspension.json", "Stop record")}</section></main></html>"""
    atomic(output / "privileged.html", text)
    for path in (output / "index.html", output / "transfer_push.html"):
        if not path.exists():
            continue
        text = path.read_text()
        start = "<!-- privileged-live-start -->"
        end = "<!-- privileged-live-end -->"
        if start in text:
            text = text[: text.index(start)] + text[text.index(end) + len(end) :]
        text = text.replace("<main>", "<main>" + start + fragment + end, 1)
        atomic(path, text)
    print(
        json.dumps(
            {
                task: {
                    k: v
                    for k, v in state.items()
                    if k in ("status", "transitions", "competence_demonstrated")
                }
                for task, state in states.items()
            }
        )
    )


if __name__ == "__main__":
    main()
