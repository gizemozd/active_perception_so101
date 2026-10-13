"""Build the live status page for the matched insertion reward runs."""

import argparse
import html
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def row(values):
    return "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    plan = read(out / "plan.json")
    runs, lines = [], []
    for case in plan["cases"]:
        condition = case["condition"]
        folder = out / condition
        state = read(folder / "state.json", {})
        submissions = [read(p) for p in sorted(folder.glob("submission-*.json"))]
        updates = state.get("updates", 0)
        run_dir = state.get("run_directory") or next(
            (str(p.parent) for p in (Path(plan["logs"]) / condition).glob("*/iterations.jsonl")),
            None,
        )
        if run_dir:
            history = Path(run_dir) / "iterations.jsonl"
            if history.exists():
                for line in reversed(history.read_text().splitlines()):
                    try:
                        updates = max(updates, int(json.loads(line)["iteration"]))
                        break
                    except (ValueError, KeyError):
                        continue
        evaluations = []
        for record in state.get("evaluations", []):
            evaluation = read(Path(record["path"]), {})
            if evaluation:
                evaluations.append(
                    dict(
                        updates=record["updates"],
                        split=record["split"],
                        occlusion=record["occlusion"],
                        successes=evaluation["successes"],
                        episodes=evaluation["episodes"],
                        per_variant=evaluation["per_variant"],
                        file=f"{condition}/{Path(record['path']).name}",
                    )
                )
        status = state.get("status", "submitted" if submissions else "prepared")
        runs.append(
            dict(
                condition=condition,
                status=status,
                updates=updates,
                transitions=updates * plan["num_envs"] * 24,
                target=plan["transitions_per_policy"],
                job_ids=[s.get("job_id") for s in submissions],
                evaluations=evaluations,
            )
        )
        scores = (
            "; ".join(
                f'<a href="{html.escape(e["file"])}">{e["split"]} {e["occlusion"]} '
                f"{e['successes']}/{e['episodes']} at update {e['updates']}</a>"
                for e in evaluations
            )
            or "Pending"
        )
        lines.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    condition,
                    status,
                    f"{updates:,}/{plan['updates']:,}",
                    f"{updates * plan['num_envs'] * 24:,}/{plan['transitions_per_policy']:,}",
                    ", ".join(str(j) for j in runs[-1]["job_ids"]),
                )
            )
            + f"<td>{scores}</td></tr>"
        )
    summary = dict(
        built_at=datetime.now(UTC).isoformat(),
        runs=runs,
        transitions_per_policy=plan["transitions_per_policy"],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Plug insertion reward camera study</title>
<style>body{{font:16px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#183944}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #cbd7d9}}a{{color:#08706a}}</style>
<h1>Plug insertion reward: matched camera policies</h1>
<p>Wrist, wrist + fixed view 7, and wrist + active camera; seed {plan["training_seed"]} and {plan["transitions_per_policy"]:,} transitions per policy. All use the original insertion distance reward, absolute XYZ controls, and the pilot's optimizer settings. Training uses clean scenes. Dynamic evaluations measure robustness.</p>
<table><thead><tr><th>Policy</th><th>Status</th><th>Updates</th><th>Transitions</th><th>Jobs</th><th>Physical evaluations</th></tr></thead><tbody>{"".join(lines)}</tbody></table>
<p>Midpoint validation: 512 episodes per setting, reset seed 30000. Final test: 1,024 episodes per setting, reset seed 80000. Success requires physical error under 2 mm for three consecutive samples. Pilot static + wrist: 2/128 at 1.23M transitions; see <a href="../plug_reward_pilot/20261009T230424Z/summary.json">pilot evidence</a>. This pilot is not a final study result.</p>
<p><a href="plan.json">Commands, source hashes and budget</a> · <a href="summary.json">Machine-readable status</a>. Source is copied byte-for-byte from the completed GPU pilot. Updated {summary["built_at"]}.</p></html>"""
    temp = out / f"index.tmp-{os.getpid()}"
    temp.write_text(document)
    temp.replace(out / "index.html")
    print(out / "index.html")


if __name__ == "__main__":
    main()
