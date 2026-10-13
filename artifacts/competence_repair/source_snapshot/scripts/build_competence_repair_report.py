"""Add independently audited competence repairs without replacing the original study."""

import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def build(root, out, read, link, table, figure, audit_evaluation):
    selection = root / "artifacts/competence_repair/report_selection.json"
    if not selection.exists():
        return "", [], {}
    chosen = read(selection)
    results, media, rows = [], [], []
    for item in chosen["results"]:
        path = root / item["report"]
        raw = read(path)
        entry = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            episodes=raw["episodes"],
            successes=raw["successes"],
        )
        d = audit_evaluation(path, entry, item["task"])
        assert d["split"] == "test" and d["seed"] == item.get("test_seed", 90000)
        checkpoint = Path(d["checkpoint"])
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if item.get("checkpoint_sha256"):
            assert checkpoint_hash == item["checkpoint_sha256"]
        result = (
            item
            | entry
            | dict(
                rate=100 * d["successes"] / d["episodes"],
                checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_hash,
                ci=[100 * x for x in d["wilson_95"]],
                seed=d["seed"],
            )
        )
        results.append(result)
        rows.append(
            [
                html.escape(item["label"]),
                f"{result['successes']}/{result['episodes']} ({result['rate']:.1f}%)",
                f"{result['ci'][0]:.1f}–{result['ci'][1]:.1f}%",
                html.escape(item["scope"]),
                link(path, "Audited test"),
            ]
        )
        if item.get("representatives"):
            records = read(root / item["representatives"])["records"]
            for record in records:
                episode = record["episode"]
                assert d["outcomes"][episode] == record["expected_success"]
                media.append(
                    dict(
                        task=item["task"],
                        condition="wrist_static",
                        seed=item.get("training_seed", 0),
                        occlusion=d["experiment"]["occlusion"],
                        outcome="success" if record["expected_success"] else "failure",
                        variant=record["variant"],
                        episode=episode,
                        caption=f"Repair · {item['label']} · {'success' if record['expected_success'] else 'failure'}",
                        outside=str(Path(os.path.relpath(root / record["outside_video"], out))),
                        policy=str(Path(os.path.relpath(root / record["policy_video"], out))),
                    )
                )
    body = '<section id="repair"><h2>Competence repairs · fresh held-out tests</h2>'
    body += "<p>Checkpoint choices used validation seed 31000. The original repair tests use seed 90000; the occlusion-trained plug model uses a separate fresh seed, 91000. Each test has 1,024 episodes and the unchanged physical success audit. These are single trained models; confidence intervals describe episode sampling, not training-seed variability.</p>"
    body += table(
        ["Model / intervention", "Success", "95% Wilson interval", "Task and method", "Evidence"],
        rows,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for task, ax in zip(["plug", "push"], axes, strict=True):
        items = [r for r in results if r["task"] == task]
        rates = np.array([r["rate"] for r in items])
        errors = np.array(
            [[r["rate"] - r["ci"][0] for r in items], [r["ci"][1] - r["rate"] for r in items]]
        )
        ax.bar(
            np.arange(len(items)),
            rates,
            yerr=errors,
            capsize=4,
            color=["#64748b", "#0f766e", "#2563eb"][: len(items)],
        )
        ax.set(
            ylim=(0, 100),
            ylabel="Held success (%)",
            title=task.capitalize() + " · fresh tests",
            xticks=np.arange(len(items)),
            xticklabels=[r.get("short_label", r["label"]) for r in items],
        )
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.2)
        for i, rate in enumerate(rates):
            ax.text(i, min(rate + 5, 97), f"{rate:.1f}%", ha="center")
        if task == "plug":
            ax.axhline(80, linestyle="--", color="#53616b", linewidth=1)
    body += figure(
        fig,
        "repair_success",
        "Fresh test outcomes. Plug uses the explicitly easier YM-only variant setting and a supervised visual estimator with a prescribed insertion motion. Push keeps the arc targets and 5 mm success rule; interventions are labeled separately. Error bars are episode-level 95% Wilson intervals.",
    )
    baseline = read(root / "artifacts/competence_repair/push_original/final-test.json")
    corrected = read(root / "artifacts/competence_repair/push_wall_ablation/final-test.json")
    assert baseline["initial_physics_state_sha256"] == corrected["initial_physics_state_sha256"]
    original_goals = np.array(
        [r["physical_goal_position"][:2] for r in baseline["termination_audit"]["records"]]
    )
    corrected_goals = np.array(
        [r["physical_goal_position"][:2] for r in corrected["termination_audit"]["records"]]
    )
    np.testing.assert_allclose(original_goals, corrected_goals, atol=1e-7, rtol=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, title in zip(
        axes, [baseline, corrected], ["Original partition", "Partition removed"], strict=True
    ):
        success = np.array(data["outcomes"], dtype=bool)
        for mask, color, label in [
            (success, "#0f766e", "Success"),
            (~success, "#c2410c", "Failure"),
        ]:
            ax.scatter(
                original_goals[mask, 0] * 1000,
                original_goals[mask, 1] * 1000,
                s=10,
                alpha=0.55,
                c=color,
                label=label,
            )
        ax.add_patch(
            Rectangle(
                (32, -17),
                8,
                40,
                facecolor="#64748b" if title == "Original partition" else "none",
                edgecolor="#334155",
                linestyle="--",
                alpha=0.7,
            )
        )
        ax.axvline(-100, color="#64748b", linestyle=":", linewidth=1)
        ax.axvline(65, color="#64748b", linestyle=":", linewidth=1)
        ax.set(title=title, xlabel="Goal X (mm)", ylabel="Goal Y (mm)", aspect="equal")
        ax.legend(fontsize=8)
    body += figure(
        fig,
        "repair_push_workspace",
        "Identical 1,024 physical resets and learned weights. Rectangle marks the old colliding partition; dotted lines mark original TCP X command limits. Removing the partition helps, but failures remain across the workspace. A goal scatter does not establish the reachability of every required pushing approach.",
    )
    body += "<h3>What changed and why</h3><p><strong>Plug:</strong> replaced difficult end-to-end policy learning with a trained RGB target estimator and a calibrated lift-align-descend controller. The deployed model reads wrist/fixed-camera images and robot proprioception; privileged target labels are used only for supervised training. This release uses YM only and an 8-second horizon. Fixture jitter remains ±8 mm and the 2 mm, three-sample hold rule is unchanged. This result does not establish competence on XM, XP, or YP.</p>"
    body += "<p><strong>Push:</strong> an obsolete colliding partition intersects the expanded arc target area. Removing only that partition improved matched validation from 346/512 to 383/512. Longer horizons (20/30 seconds) and fine-tuning with lower action noise did not improve validation. The visual occluder, target distribution, 5 mm tolerance, settling, contact-free and tool-withdrawal criteria remain. Retraining and any widened action bounds are stated in the table. Large out-of-range action means are an additional optimization concern; the regularization trial is reported only through its measured outcomes.</p>"
    failures = read(root / "artifacts/competence_repair/push_remaining_failures.json")
    body += f"<p>In the selected push policy's corrected-environment test, all {failures['failures']} failures still missed the 5 mm placement threshold; median failed XY error was {failures['median_failure_xy_mm']:.1f} mm. {failures['outside_goal_failures']} failures had goal X outside the original command range. The low score is therefore primarily a placement problem, rather than successful placements rejected solely for withdrawal or settling. Expanding the command range requires separate validation; the tested PPO fine-tunes did not improve the selected result.</p>"
    body += "<p><strong>Remaining limits:</strong> the plug repairs use wrist plus a fixed camera and YM only. The push population check spans all three camera conditions. These results do not demonstrate an active-camera advantage or replace the original three-seed comparison below. Multi-variant plug prototypes and unsuccessful controls are retained in the repair log.</p>"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for curve in chosen.get("validation_curves", []):
        points = []
        for step, relative in curve["points"]:
            d = read(root / relative)
            assert d["split"] == "validation"
            points.append((step, 100 * d["successes"] / d["episodes"]))
        if points:
            x, y = zip(*points, strict=True)
            axes[["plug", "push"].index(curve["task"])].plot(x, y, marker="o", label=curve["label"])
    axes[1].axhline(
        100 * 383 / 512, color="#53616b", linestyle="--", label="Frozen partition-removal control"
    )
    for task, ax in zip(["plug", "push"], axes, strict=True):
        ax.set(
            title=task.capitalize() + " · validation checkpoints",
            xlabel="Training step / PPO update",
            ylabel="Held success (%)",
            ylim=(0, 100),
        )
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=8)
    body += figure(
        fig,
        "repair_validation",
        "Validation curves show checkpoint selection and unstable learning, including regressions. Final-test outcomes above were not used for checkpoint selection. Supervised steps and PPO updates are different training units.",
    )
    controls = []
    for name, relative in [
        ("Original frozen policy · 12s", "push_horizon/12/evaluation.json"),
        ("Frozen policy · 20s", "push_horizon/20/evaluation.json"),
        ("Frozen policy · 30s", "push_horizon/30/evaluation.json"),
        ("Partition removal only · selected", "push_wall_ablation/evaluation.json"),
        ("Lower-noise fine-tune · std .3 · 250", "push_noise03/intermediate-250.json"),
        ("Lower-noise fine-tune · std 1 · 250", "push_noise10/intermediate-250.json"),
        ("Clear scene · fine-tune 250", "push_clear_v2/intermediate-250.json"),
        ("Clear scene · fine-tune 500", "push_clear_v2/intermediate-500.json"),
        ("Clear + wide bounds · fine-tune 250", "push_clear_wide_v2/intermediate-250.json"),
        ("Clear + wide bounds · fine-tune 500", "push_clear_wide_v2/intermediate-500.json"),
        ("Clear + wide + action regularization · 250", "push_regularized/intermediate-250.json"),
        ("Frozen policy · wide bounds directly", "push_frozen_wide_direct/validation.json"),
        ("Frozen policy · wide bounds + affine remap", "push_frozen_wide_remap/validation.json"),
    ]:
        path = root / "artifacts/competence_repair" / relative
        data = read(path)
        assert data["split"] == "validation" and data["seed"] == 31000
        entry = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            episodes=data["episodes"],
            successes=data["successes"],
        )
        audit_evaluation(path, entry, "push")
        controls.append(
            [
                name,
                f"{data['successes']}/{data['episodes']} ({100 * data['success_rate']:.1f}%)",
                link(path, "Raw validation"),
            ]
        )
    body += (
        "<details><summary>Push control experiments and unsuccessful fine-tunes</summary><p>These use validation seed31000. They informed selection; the separate final tests are shown above.</p>"
        + table(["Control", "Success", "Evidence"], controls)
        + "</details>"
    )
    selected = read(root / "artifacts/competence_repair/selected_models.json")
    for model in selected:
        assert (
            hashlib.sha256((root / model["checkpoint"]).read_bytes()).hexdigest() == model["sha256"]
        )
    body += (
        "<p>"
        + link(
            root / "artifacts/competence_repair/selected_models.json",
            "Deployable checkpoints and exact evaluation commands",
        )
        + " · "
        + link(
            root / "artifacts/competence_repair/source_snapshot/manifest.json",
            "Reproduction source hashes",
        )
        + "</p>"
    )
    body += (
        "<p>"
        + link(selection, "Frozen selection and reproduction details")
        + ' · <a href="repair_analysis.json">Repair analysis and hashes</a> · <a href="#videos">Recorded success and failure examples</a></p></section>'
    )
    from build_push_population_report import build as build_population

    population_body, population = build_population(
        root, read, link, table, figure, audit_evaluation
    )
    body = body.replace("<h3>What changed", population_body + "<h3>What changed")
    summary = dict(
        results=results,
        push_population=population,
        plots=["repair_success", "repair_validation", "repair_push_workspace"],
        selection=str(selection.relative_to(root)),
        media_examples=len(media),
    )
    if population:
        summary["plots"].append("repair_push_population")
    (out / "repair_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    return body, media, summary
