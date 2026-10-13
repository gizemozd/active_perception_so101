"""Build a concise local report from the saved common-control study evidence."""

import csv
import html
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path


def plot_results(root, means, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    for stage in ("control_v2", "camera_v1"):
        groups = [g for g in means if g["stage"] == stage and "clean_mean" in g]
        if not groups:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, max(3, len(groups) * 0.42)), sharey=True)
        for ax, mode in zip(axes, ("clean", "dynamic"), strict=True):
            for index, group in enumerate(groups):
                if mode + "_mean" not in group:
                    continue
                value = 100 * group[mode + "_mean"]
                ax.barh(index, value, color="#287591", alpha=0.65)
                values = [
                    100 * r[mode]
                    for r in rows
                    if r["stage"] == stage
                    and r["case"].rsplit("_s", 1)[0] == group["recipe"]
                    and mode in r
                ]
                ax.scatter(values, [index] * len(values), color="#16344a", s=20, zorder=3)
                ax.text(value + 0.4, index, f"{value:.2f}%", va="center", fontsize=9)
            maximum = max(
                (100 * r[mode] for r in rows if r["stage"] == stage and mode in r), default=0
            )
            ax.set_xlim(0, max(5, maximum * 1.3 + 1))
            ax.set_xlabel("Insertion success (%)")
            ax.set_title(mode.capitalize())
            ax.grid(axis="x", alpha=0.2)
        axes[0].set_yticks(range(len(groups)), [g["recipe"] for g in groups])
        axes[0].invert_yaxis()
        fig.suptitle("Small-policy screen: bars = mean, dots = training seeds")
        fig.tight_layout()
        output = root / "figures" / stage
        output.parent.mkdir(exist_ok=True)
        for extension in ("svg", "png", "pdf"):
            fig.savefig(output.with_suffix("." + extension), facecolor="white", dpi=160)
        plt.close(fig)
        figures.append(
            f'<figure><img src="figures/{stage}.svg" style="max-width:100%" alt="{stage} success by recipe and training seed"><figcaption>Notice the plotted scale: these are short training runs, not converged policies. <a href="figures/{stage}.pdf">PDF</a></figcaption></figure>'
        )
    return "".join(figures)


def read_evaluation(path):
    data = json.loads(path.read_text())
    assert len(data["outcomes"]) == data["episodes"]
    assert sum(data["outcomes"]) == data["successes"]
    audit = data["termination_audit"]
    for key, value in audit.items():
        if key.endswith("violations") and value is not None:
            assert value == 0, (path, key, value)
    for record in audit["records"]:
        if record["success"]:
            assert record["success_hold_steps"] >= 3
            assert max(record["last_three_manager_errors_m"]) < 0.002
    return data


def main():
    root = Path(__file__).resolve().parents[1] / "artifacts/common_control_study"
    rows = []
    planned_cases = set()
    for path in sorted(root.glob("*/plan.json")):
        plan = json.loads(path.read_text())
        if not (path.parent / "training_submission.json").exists():
            continue
        planned_cases.update((path.parent.name, case["name"]) for case in plan["cases"])
        for case in plan["cases"]:
            directory = path.parent / case["name"]
            cfg = case["prepared"]["experiment"]
            row = dict(
                stage=path.parent.name,
                case=case["name"],
                condition=cfg["condition"],
                seed=cfg["seed"],
                control=cfg["translation_control"],
                distribution=case["prepared"]["action_distribution"],
                camera_interval=cfg["camera_interval"],
                training_occlusion=cfg["occlusion"],
                camera_start=cfg.get("camera_start", "home"),
                camera_workspace=cfg.get("camera_workspace", "local_v1"),
                budget=plan["transitions_per_policy"],
                completed=(directory / "completed.json").exists(),
            )
            training = directory / "training.json"
            if training.exists():
                data = json.loads(training.read_text())
                row["actual_transitions"] = data["actual_transitions"]
                assert row["actual_transitions"] == row["budget"]
            for mode in ("clean", "dynamic"):
                report = directory / f"test-{mode}.json"
                if report.exists():
                    data = read_evaluation(report)
                    row[mode] = data["success_rate"]
                    row[mode + "_successes"] = data["successes"]
                    row[mode + "_episodes"] = data["episodes"]
                    row[mode + "_source"] = str(report.relative_to(root))
                    row[mode + "_terminal_error_mm"] = (
                        1000 * data["termination_audit"]["mean_terminal_position_error_m"]
                    )
                    row[mode + "_per_variant"] = data["per_variant"]
                    errors = [
                        1000 * r["qpos_position_error_m"]
                        for r in data["termination_audit"]["records"]
                        if not r["success"]
                    ]
                    row[mode + "_failure_median_error_mm"] = (
                        statistics.median(errors) if errors else None
                    )
                    row[mode + "_failures_over_20mm"] = sum(e > 20 for e in errors)
            row["interventions"] = {}
            for intervention_path in directory.glob("intervention-*.json"):
                data = read_evaluation(intervention_path)
                normal = read_evaluation(directory / "test-dynamic.json")
                assert (
                    data["initial_physics_state_sha256"] == normal["initial_physics_state_sha256"]
                )
                row["interventions"][intervention_path.stem.removeprefix("intervention-")] = {
                    "successes": data["successes"],
                    "episodes": data["episodes"],
                    "success_rate": data["success_rate"],
                    "source": str(intervention_path.relative_to(root)),
                }
            selection = directory / "selection.json"
            if selection.exists():
                row["selection"] = json.loads(selection.read_text())
            rows.append(row)
    assert {(row["stage"], row["case"]) for row in rows} == planned_cases
    assert len(rows) == len(planned_cases)
    groups = {}
    for row in rows:
        key = (row["stage"], row["case"].rsplit("_s", 1)[0])
        groups.setdefault(key, []).append(row)
    means = []
    for (stage, name), group in groups.items():
        item = dict(stage=stage, recipe=name)
        for mode in ("clean", "dynamic"):
            values = [r[mode] for r in group if mode in r]
            if values:
                item[mode + "_mean"] = statistics.mean(values)
                item[mode + "_seed_sd"] = statistics.stdev(values) if len(values) > 1 else None
                item[mode + "_seeds"] = len(values)
        means.append(item)
    pairs = []
    by_case = {(r["stage"], r["case"]): r for r in rows}
    for row in rows:
        comparison = None
        if row["stage"] == "control_v2" and row["control"] == "delta":
            comparison = row["case"].replace("delta_", "absolute_", 1)
        elif row["stage"] == "camera_v1" and row["condition"] == "active":
            comparison = row["case"].replace("_active_", "_wrist_static_")
            comparison = comparison.replace("inspection_slow_", "inspection_dynamic_")
        baseline = by_case.get((row["stage"], comparison))
        if baseline is None:
            continue
        for mode in ("clean", "dynamic"):
            if mode not in row or mode not in baseline:
                continue
            a, b = [read_evaluation(root / r[mode + "_source"]) for r in (baseline, row)]
            assert a["initial_physics_state_sha256"] == b["initial_physics_state_sha256"]
            pairs.append(
                dict(
                    stage=row["stage"],
                    baseline=baseline["case"],
                    candidate=row["case"],
                    mode=mode,
                    difference_pp=100 * (row[mode] - baseline[mode]),
                    initial_physics_match=True,
                )
            )
    result = dict(
        updated_utc=datetime.now(timezone.utc).isoformat(),
        scope="Small-budget screening; two training seeds. Episode counts do not replace independent training seeds. Final tests excluded from checkpoint selection.",
        rows=rows,
        means=means,
        paired_comparisons=pairs,
    )
    conclusions = ""
    if rows and all(row["completed"] for row in rows):
        slow = [
            r
            for r in rows
            if r["stage"] == "camera_v1" and r["case"].startswith("inspection_slow_")
        ]
        fixed = [
            r
            for r in rows
            if r["stage"] == "camera_v1"
            and r["case"].startswith("inspection_dynamic_wrist_static_")
        ]
        active_rate = 100 * statistics.mean(r["dynamic"] for r in slow)
        fixed_rate = 100 * statistics.mean(r["dynamic"] for r in fixed)
        held_rate = 100 * statistics.mean(
            r["interventions"]["hold_external_0"]["success_rate"] for r in slow
        )
        conclusions = f'<section><h2>Completed screen: controls unified, active-perception benefit remains unproven</h2><p>All {len(rows)} fresh policies completed 1,228,800 transitions each. Delta alone gave a small gain for active policies (0% to 1.76% clean success), but no general boost across sensing conditions. These small from-scratch runs are not comparable to the longer-trained original benchmark or the specialized YM-only repair.</p><p>The camera follow-ups remain at low absolute success. The slower-camera recipe reaches {active_rate:.2f}% on dynamic tests versus {fixed_rate:.2f}% for its fixed-camera match; holding the external image from reset reaches {held_rate:.2f}%. The raw active-policy gain therefore does not establish a benefit from updated external views in this screen.</p><p>The sampled physical controller can solve the task and better views are reachable, but the learned policies remain poor at the four-variant manipulation task. Establishing reliable visual estimation and manipulation competence is the next priority before a stronger claim about active sensing. The initial clear interval and fixed target also reduce the need for later camera motion.</p><p><a href="verification.json">Budget, selection, source and physical-outcome verification</a></p></section>'
        result["conclusion"] = {
            "active_perception_benefit_established": False,
            "slow_camera_dynamic_percent": active_rate,
            "matched_fixed_dynamic_percent": fixed_rate,
            "slow_camera_held_image_dynamic_percent": held_rate,
            "scope": "Two training seeds per recipe; 1,228,800 transitions per fresh policy; original four-variant 3.5-second plug task.",
        }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = ["stage", "case", "budget", "completed", "clean", "dynamic"]
    with (root / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    body = []
    for row in rows:
        cells = [
            html.escape(row["stage"]),
            html.escape(row["case"]),
            f"{row['budget']:,}",
            "Complete" if row["completed"] else "Pending / running",
        ]
        for mode in ("clean", "dynamic"):
            cells.append(
                f'<a href="{row[mode + "_source"]}">{row[mode + "_successes"]}/{row[mode + "_episodes"]} ({100 * row[mode]:.1f}%)</a>'
                if mode in row
                else "—"
            )
        body.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
    aggregate = []
    for group in means:
        cells = [html.escape(group["stage"]), html.escape(group["recipe"])]
        for mode in ("clean", "dynamic"):
            count = group.get(mode + "_seeds", 0)
            if count:
                mean = 100 * group[mode + "_mean"]
                sd = group[mode + "_seed_sd"]
                cells.append(
                    f"{mean:.2f}%"
                    + (f" ± {100 * sd:.2f}" if sd is not None else "")
                    + f" ({count}/2 seeds)"
                )
            else:
                cells.append("Pending")
        aggregate.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
    page = '<!doctype html><meta charset="utf-8"><title>SO-101 common control study</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;background:white;color:#182433}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}a{color:#145baa}</style><h1>Common delta controls and active-camera screening</h1>'
    page += "<p>Shared manipulator XYZ and camera XYZ/pan/tilt target increments at 25 Hz. Fixed and active conditions start with the same arm-mounted camera. Original four plug variants, 2 mm / three-sample physical success, 3.5-second horizon and unchanged insertion reward.</p><p>Control screen: absolute and delta commands share a 5 mm per-step slew limit. Camera follow-ups separately test bounded exploration, slower camera decisions and dynamic-occlusion training. Each recipe uses two training seeds; this is a small-budget screen.</p>"
    page += "<p>All camera-stage recipes use the expanded inspection workspace, initial action noise 0.3 and entropy coefficient 0.003. Compare recipes within that stage to isolate the distribution, starting pose, training occlusion or camera interval; a direct comparison against the control stage also changes exploration settings and camera bounds.</p>"
    page += conclusions
    page += plot_results(root, means, rows)
    page += (
        "<h2>Across training seeds</h2><p>Mean success ± sample SD between training seeds. Two seeds and a short training budget do not establish convergence or statistical significance.</p><table><tr><th>Stage</th><th>Recipe</th><th>Clean</th><th>Dynamic</th></tr>"
        + "".join(aggregate)
        + "</table><h2>Individual runs</h2>"
    )
    page += (
        '<p><a href="summary.json">Machine-readable summary</a> · <a href="results.csv">CSV</a> · <a href="../training_results/index.html">Original report</a></p><table><thead><tr><th>Stage</th><th>Policy</th><th>Transitions</th><th>Status</th><th>Clean test</th><th>Dynamic test</th></tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table>"
    )
    page += '<p>Inspection recipes start both fixed and active cameras at a calibrated reachable pose. In the independent geometric audit, plug landmark visibility rises from 8/32 at the original home to 31/32 at this pose. Per-reset view selection also reaches 31/32: this supports better placement, not an adaptive-camera advantage.</p><p><a href="view_audit/inspection_v2_realized/results.json">Calibrated view audit</a></p>'
    page += '<p>The privileged scripted controller completes 32/32 native-MuJoCo insertions with each matched control mapping. This establishes sampled physical feasibility, not learned policy competence. <a href="manipulation_feasibility.json">Physical feasibility audit</a></p><img src="view_audit/camera_views.png" alt="Wrist, original external and calibrated inspection views for each plug variant" style="max-width:100%">'
    dynamic_audit = root / "view_audit/inspection_v2_realized_dynamic/results.json"
    if dynamic_audit.exists():
        data = json.loads(dynamic_audit.read_text())["tasks"]["plug"]["summary"]
        page += f'<p>With the manipulation scene held at reset and the existing blocker probed at five sweep phases, the best tested fixed view exposes the landmarks in {data["best_single_endpoint"]}/{data["episodes"]} sampled scenes; an oracle choosing among the tested valid views reaches {data["per_reset_oracle"]}/{data["episodes"]}. This suggests some adaptive information opportunity during occlusion. It does not include camera travel time or demonstrate insertion success. <a href="{dynamic_audit.relative_to(root)}">Sweep-phase visibility audit</a></p>'
    page += '<p>The existing plug sweep starts 0.35–1.47 seconds into the episode and clears by 2.59 seconds at the latest. The fixture stays fixed. Thus an initially clear view, recurrent memory and later clearance can substitute for camera motion; active sensing is useful only when it resolves remaining uncertainty in time. This is a task-structure inference, not a claim that movement can never help. <a href="information_opportunity.json">Timing and information audit</a></p>'
    page += "<h2>Does the active policy use updated camera images?</h2><p>Matched dynamic tests freeze the external RGB stream at reset or after the inspection second. Physically freezing the camera also changes motion and proprioception; it is not a pure visual intervention.</p><table><tr><th>Policy</th><th>Normal</th><th>Hold image at reset</th><th>Hold image after 1 s</th><th>Hold camera after 1 s</th></tr>"
    for row in rows:
        if not row["interventions"]:
            continue
        cells = [html.escape(row["case"]), f"{100 * row['dynamic']:.2f}%"]
        for label in ("hold_external_0", "hold_external_1", "freeze_camera_1"):
            data = row["interventions"].get(label)
            cells.append(
                f'<a href="{data["source"]}">{100 * data["success_rate"]:.2f}%</a>'
                if data
                else "Pending"
            )
        page += "<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>"
    page += "</table>"
    page += '<p>All available paired absolute/delta and fixed/active comparisons have identical initial physical-state hashes. Raw successes are checked against episode flags and the physical three-sample hold criterion. <a href="control_contract.json">Shared action-space contract</a></p>'
    page += "<p>Updated " + result["updated_utc"] + "</p>"
    (root / "index.html").write_text(page)
    original = root.parent / "training_results/index.html"
    if original.exists():
        complete = sum(r["completed"] for r in rows)
        note = f'<section id="common-control-study"><h2>New: shared delta controls and camera experiments</h2><p>{complete}/{len(rows)} small-policy runs complete. Plug and push now have a shared, versioned XYZ manipulator / XYZ-pan-tilt camera delta interface. The original checkpoint mappings remain available.</p><p>The completed two-seed control screen does not show a general insertion improvement from delta commands: active policies reach 1.76% clean and 0.59% dynamic success, while wrist-only and fixed-camera results remain near zero. Matched camera follow-ups separately test bounded exploration, improved placement, occlusion training and slower decisions.</p><p><a href="../common_control_study/index.html">Open the live local study report, comparisons and diagnostic evidence</a>. Updated {result["updated_utc"]}. The original benchmark and competence repairs below have different budgets and scopes.</p></section>'
        if conclusions:
            note = note.replace(
                "</section>",
                f"<p>Camera follow-up outcome: slower-camera dynamic success {active_rate:.2f}%, matched fixed {fixed_rate:.2f}%, external image held from reset {held_rate:.2f}%. Updated external views have not demonstrated a benefit in this small-policy screen.</p></section>",
            )
        document = original.read_text()
        if 'id="common-control-study"' in document:
            document = re.sub(
                r'<section id="common-control-study">.*?</section>',
                lambda _: note,
                document,
                count=1,
                flags=re.DOTALL,
            )
        else:
            document = document.replace("<main>", "<main>" + note, 1)
        original.write_text(document)
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
