"""Summarize all nine push arc camera runs without mixing older studies."""

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    folder = args.root / "artifacts/push_rgb_arc120"
    plan = json.loads((folder / "study-plan.json").read_text())
    rows = []
    summary = {"updated_utc": datetime.now(timezone.utc).isoformat(), "runs": []}
    for run in plan["runs"]:
        run_folder = folder / "runs" / run["id"]
        state_path = run_folder / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        submissions = [json.loads(p.read_text()) for p in sorted(run_folder.glob("submission-*.json"))]
        evaluations = state.get("evaluations", [])
        log_path = state.get("run_directory")
        observed = state.get("transitions", 0)
        if log_path:
            iterations = Path(log_path) / "iterations.jsonl"
            if iterations.exists():
                for line in reversed(iterations.read_text(errors="replace").splitlines()):
                    try:
                        observed = max(observed, json.loads(line)["transitions"])
                        break
                    except (ValueError, KeyError):
                        pass
        summary["runs"].append(dict(run=run, status=state.get("status", "submitted"),
                                    observed_transitions=observed, evaluations=evaluations,
                                    submissions=submissions))
        latest = []
        for occlusion in ("clean", "dynamic"):
            values = [e for e in evaluations if e["occlusion"] == occlusion]
            if values:
                e = max(values, key=lambda item: item["transitions"])
                latest.append(f"{occlusion}: {e['successes']}/{e['episodes']} at {e['transitions']:,}")
        jobs = ", ".join(str(s.get("job_id", "pending")) for s in submissions)
        rows.append("<tr>" + "".join(
            f"<td>{html.escape(str(value))}</td>" for value in (
                run["condition"], run["seed"], state.get("status", "submitted"),
                f"{observed:,} / {plan['rounded_transitions']:,}", "; ".join(latest) or "pending",
                jobs or "pending",
            )
        ) + "</tr>")
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Push arc RGB study</title>
<style>body{{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#183244}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #cad5df;text-align:left}}th{{background:#eef3f7}}img{{max-width:100%}}</style>
<h1>Push arc RGB study</h1><p>Updated {summary['updated_utc']}. Three camera conditions, three independent training seeds each. Every run targets {plan['rounded_transitions']:,} transitions with a 50–120 mm object-relative target and ±80° angle.</p>
<p>At reset, the fixed camera saw the target marker in 68/512 held-out layouts. Some far targets exceed the current manipulator's action range; interpret absolute success rates with this limitation.</p>
<table><thead><tr><th>Condition</th><th>Seed</th><th>Status</th><th>Transitions</th><th>Latest held-out scores</th><th>Slurm jobs</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p><a href="../push_rgb/arc_initial_conditions/fixed-camera-visibility.json">Visibility audit</a> · <a href="../push_rgb/arc_initial_conditions/reachability.json">Reachability screen</a> · <a href="study-plan.json">Study plan</a></p></html>"""
    (folder / "index.html").write_text(body)


if __name__ == "__main__":
    main()
