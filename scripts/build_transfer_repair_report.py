"""Build the live privileged transfer reward-repair diagnostic page."""

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    base = root / "artifacts/transfer_repair"
    output = root / "artifacts/training_results"
    plan = read(base / "plan.json", {})
    state = read(base / "transfer/runtime.json", {})
    rows = []
    for ev in state.get("evaluations", []):
        rows.append(
            f"<tr><td>{ev['transitions']:,}</td><td>{ev['seed']}</td>"
            f"<td>{ev['successes']}/{ev['episodes']} ({ev['success_rate']:.1%})</td>"
            f"<td>{html.escape(json.dumps(ev.get('stages', {}), sort_keys=True))}</td>"
            f"<td><a href='../transfer_repair/transfer/{html.escape(Path(ev['path']).name)}'>Raw evaluation</a></td></tr>"
        )
    figure = ""
    run_dir = state.get("run_directory")
    if run_dir:
        path = Path(run_dir) / "iterations.jsonl"
        if path.exists():
            history = []
            for line in path.read_text().splitlines():
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    break
            if history:
                stride = max(1, len(history) // 2000)
                history = history[::stride]
                fig, axes = plt.subplots(figsize=(8, 4))
                axes.plot(
                    [r.get("transitions", 0) / 1e6 for r in history],
                    [r.get("train/Train/mean_reward", math.nan) for r in history],
                    label="Training episode return",
                )
                axes.set_xlabel("Million transitions")
                axes.grid(alpha=0.2)
                axes.legend()
                output.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    output / "transfer_repair_learning.svg", format="svg", bbox_inches="tight"
                )
                plt.close(fig)
                figure = '<figure><img style="max-width:100%" src="transfer_repair_learning.svg" alt="Transfer return curve"></figure>'
    goal = 1_000_000_000
    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transfer reward repair</title><style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1000px;margin:36px auto;padding:0 20px;color:#183244}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ccd7df;text-align:left}}th{{background:#edf3f7}}.callout{{background:#fff7e8;border-left:4px solid #d97706;padding:16px}}</style></head><body>
<p><a href="index.html">Training results index</a></p><h1>Transfer: repair the transport learning signal</h1>
<p>This state-only diagnostic follows the 20M transfer run that lifted the cube on about 99.5% of held-out episodes but never transported, placed or released it. That run used a unit-amplitude distance reward, which yielded little incentive to move a lifted cube away from the source.</p>
<div class="callout">The new opt-in <code>transfer_transport</code> reward raises the lifted transport term to <code>2 + 6 exp(-distance / 0.06)</code>. This isolates the observed failure stage; physical success criteria, action bounds, task geometry, and ±20 mm reset jitter remain unchanged. No RGB run should use this revision unless privileged evaluation shows transport learning.</div>
<p>Status: <b>{html.escape(state.get("status", "prepared"))}</b> · {state.get("transitions", 0):,} / {math.ceil(goal / (plan.get("num_envs", 2048) * 24)) * plan.get("num_envs", 2048) * 24:,} transitions · Slurm job {html.escape(str(state.get("job_id", "—")))}.</p>
{figure}<h2>Held-out privileged evaluations</h2><table><thead><tr><th>Transitions</th><th>Seed</th><th>Physical success</th><th>Failure stage counts</th><th>Data</th></tr></thead><tbody>{"".join(rows) if rows else '<tr><td colspan="5">Pending</td></tr>'}</tbody></table>
<p>The diagnostic uses 512 episodes at each of 2M, 5M and 10M transitions, on two held-out seeds. It is not an RGB camera comparison. If the new reward still produces no transport, the next experiment must change the task/control curriculum before the 1B visual runs.</p>
<p><a href="../transfer_repair/plan.json">Frozen diagnostic plan</a></p></body></html>"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "transfer_repair.html").write_text(html_doc)
    summary = {
        "status": state.get("status", "prepared"),
        "transitions": state.get("transitions", 0),
        "evaluations": state.get("evaluations", []),
        "competence_demonstrated": state.get("competence_demonstrated"),
    }
    (output / "transfer_repair_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    index = output / "index.html"
    if index.exists():
        text = index.read_text()
        link = '<a href="transfer_repair.html">Transfer reward repair</a>'
        if link not in text:
            text = text.replace("<nav>", "<nav>" + link, 1)
            index.write_text(text)


if __name__ == "__main__":
    main()
