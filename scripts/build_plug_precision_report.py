#!/usr/bin/env python3
"""Live HTML status/results for the bounded plug precision experiment."""

import argparse
import html
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def row(cells):
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    plan = read(out / "plan.json")
    progress, measurements, failures, cases = [], [], [], []
    for case in plan["cases"]:
        condition = case["condition"]
        folder = out / condition
        state = read(folder / "state.json", {})
        submissions = [read(p) for p in sorted(folder.glob("submission-*.json"))]
        status = state.get("status", submissions[-1]["status"] if submissions else "prepared")
        updates = state.get("updates", 0)
        run_dirs = list((Path(plan["logs"]) / condition).glob("*/iterations.jsonl"))
        if run_dirs:
            lines = run_dirs[0].read_text().splitlines()
            for line in reversed(lines):
                try:
                    entry = json.loads(line)
                    updates = max(updates, int(entry["transitions"]) // (plan["num_envs"] * 24))
                    break
                except (ValueError, KeyError):
                    continue
        transitions = updates * plan["num_envs"] * 24
        jobs = ", ".join(str(s.get("job_id", "—")) for s in submissions)
        progress.append(
            row(
                [
                    condition,
                    html.escape(status),
                    jobs,
                    f"{updates:,}/{plan['updates']:,}",
                    f"{transitions:,}/{plan['transitions_per_policy']:,}",
                ]
            )
        )
        records = []
        for path in sorted(folder.glob("evaluation-*.json")):
            data = read(path)
            assert data["successes"] == sum(data["outcomes"])
            audit = data["termination_audit"]
            assert len(audit["records"]) == data["episodes"]
            variants = data["per_variant"]
            label = f'<a href="{condition}/{path.name}">{data["split"]} · {data["experiment"]["occlusion"]}</a>'
            count = f"{data['successes']}/{data['episodes']} ({data['success_rate']:.1%})"
            measurements.append(
                row(
                    [
                        condition,
                        label,
                        f"model_{Path(data['checkpoint']).stem.split('_')[-1]}",
                        count,
                        *[
                            f"{variants[v]['successes']}/{variants[v]['episodes']}"
                            for v in ("xm", "xp", "ym", "yp")
                        ],
                    ]
                )
            )
            failed = [r for r in audit["records"] if not r["success"]]
            stage_counts = [
                sum(not r["plug_raised_history"] for r in failed),
                sum(r["plug_raised_history"] and not r["plug_aligned_history"] for r in failed),
                sum(r["plug_aligned_history"] for r in failed),
            ]
            assert sum(stage_counts) == len(failed)
            failures.append(row([condition, label, *stage_counts, len(failed)]))
            records.append(
                dict(
                    path=str(path),
                    split=data["split"],
                    occlusion=data["experiment"]["occlusion"],
                    successes=data["successes"],
                    episodes=data["episodes"],
                    per_variant=variants,
                )
            )
        cases.append(
            dict(
                condition=condition,
                status=status,
                jobs=jobs,
                updates=updates,
                transitions=transitions,
                evaluations=records,
            )
        )
    built = datetime.now(UTC).isoformat()
    summary = dict(
        built_at=built, transitions_per_policy=plan["transitions_per_policy"], cases=cases
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    evaluation_table = (
        table(
            ["Policy", "Evaluation", "Checkpoint", "Overall", "XM", "XP", "YM", "YP"], measurements
        )
        if measurements
        else "<p>No learned evaluation has finished yet. Success rates are pending.</p>"
    )
    failure_table = (
        table(
            [
                "Policy",
                "Evaluation",
                "No lift history",
                "Lifted, never aligned",
                "Aligned, insertion failed",
                "All failures",
            ],
            failures,
        )
        if failures
        else "<p>Stage failure counts will appear after evaluation.</p>"
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Plug precision training · 18M camera comparison</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#edf3f2;color:#183944;margin:0}}main{{max-width:1200px;margin:auto;padding:30px 20px 70px}}header{{padding:30px;background:#153f46;color:white;border-radius:16px}}h1{{line-height:1.15;font-size:2.8rem}}section{{padding:22px 28px;margin:20px 0;border:1px solid #cbd7d9;border-radius:12px;background:white}}a{{color:#08706a}}header a{{color:#b7ebdb}}table{{border-collapse:collapse;width:100%;font-size:.92rem}}td,th{{padding:11px;text-align:left;border-bottom:1px solid #d1dedf}}th{{background:#eaf2f0}}.scroll{{overflow-x:auto}}li{{margin:10px 0}}.note{{border-left:4px solid #bb8524;padding:12px;background:#fff5df}}small{{color:#536970}}</style></head><body><main><header><div>SO101 · PLUG · MATCHED SEED-0 EXPERIMENT</div><h1>Learn the approach.<br>Control the final millimeters.</h1><p>Wrist-only, static+wrist and active+wrist; {plan["transitions_per_policy"]:,} transitions each, {plan["total_training_transitions"]:,} total. Training uses clean scenes; dynamic occlusion is an evaluation of robustness to an unseen obstruction.</p><a href="../xm_diagnosis/index.html">XM diagnosis</a> · <a href="../training_results/index.html">Earlier results and hyperparameter search</a></header>
<section><h2>Training status</h2>{table(["Policy", "Status", "Slurm jobs", "PPO updates", "Transitions"], progress)}<p>Each policy uses512 worlds ×24 steps/update,1465 updates,seed0. Static uses the previously selected grazing view7. The midpoint is750 updates /9,216,000 transitions. Both allocations and source snapshots are recorded.</p></section>
<section><h2>What changed</h2><ul><li><b>Incremental control:</b> each XYZ action requests at most5 mm per axis per40 ms step. A zero command holds the previous target. Workspace limits and the1-second inspection pause remain.</li><li><b>Exploration:</b> initial normalized standard deviation0.25 gives1.25 mm XYZ request noise per axis. Entropy coefficient0.001, adaptive initial learning rate0.0003. Standard deviation is learned; the initial value is not a permanent cap.</li><li><b>Stage reward:</b> physical TCP lift above87 mm, then XY alignment within3 mm while TCP remains above84 mm, then insertion. Stage density lies in[0,3]; reward is0.2×density +100×held-success −0.005 −0.001×mean squared manipulator action. Current XY/Z error still determines precision credit after alignment.</li><li><b>Inputs:</b> actors receive their specified RGB views and measurable robot state. True object/goal positions and reward-stage history are available only to reward/critic. All actors use a compact GRU.</li></ul><p>All three policies use the same three XYZ controls. Physical success is the unchanged body distance&lt;2 mm for three consecutive samples, within the original3.5-second horizon. No yaw action or extra orientation criterion is enabled in this comparison.</p></section>
<section><h2>Measured learned-policy results</h2>{evaluation_table}<p>Midpoint validation uses512 episodes per setting at seed30000. Final tests use1024 episodes per setting at seed80000; every variant has256 final-test episodes. All evaluations use deterministic actors and audit physical state before reset. One training seed does not establish a general camera ranking.</p></section>
<section><h2>Failure stages</h2>{failure_table}<p>Stage history is read before termination/reset, through the previous reward sample. It diagnoses where an episode stopped progressing; actual success is scored independently from physical position.</p></section>
<section><h2>Validation and reproducibility</h2><div class="note"><b>512/512 is scripted feasibility, not trained performance.</b> A privileged lift-first controller passed all512 randomized GPU episodes (128 of each variant) under the new controls. Each camera condition then passed two real rendered PPO updates, encoder updates and checkpoint reload checks.</div><p><a href="gpu-validation.json">GPU feasibility and PPO smoke evidence</a> · <a href="audit-submission.json">GPU audit source hashes</a> · <a href="plan.json">Exact training commands, budget and source hashes</a> · <a href="summary.json">Machine-readable status</a></p><p>Training runs from a frozen copy of the GPU-tested source. Two explicitly recorded compatibility edits default missing reward-profile attributes in legacy diagnostic stubs; they do not change serialized Experiment behavior. The midpoint resume preserves weights, Adam, adaptive LR, update count and W&amp;B identity; simulator/RNG/recurrent episode state resets on resume.</p><small>Built {built}. Workers refresh this page after evaluations. Rebuild manually with scripts/build_plug_precision_report.py --output artifacts/plug_precision.</small></section></main></body></html>"""
    temp = out / f"index.tmp-{os.getpid()}"
    if (out / "analysis.html").exists():
        document = document.replace(
            "<section><h2>Measured learned-policy results</h2>",
            '<section><h2>Completed-run analysis</h2><p><a href="analysis.html">'
            "Final failure stages, terminal errors, learning curves and trajectory videos"
            "</a></p></section><section><h2>Measured learned-policy results</h2>",
        )
    temp.write_text(document)
    temp.replace(out / "index.html")
    print(out / "index.html")


if __name__ == "__main__":
    main()
