"""Audit and analyze completed precision runs using saved evaluations and traces."""

import argparse
import csv
import hashlib
import html
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(headers, rows):
    def cells(tag, values):
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values) + "</tr>"

    return "<table>" + cells("th", headers) + "".join(cells("td", r) for r in rows) + "</table>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/plug_precision"))
    args = parser.parse_args()
    out = args.output.resolve()
    plan = read(out / "plan.json")
    for relative, expected in plan["source_sha256"].items():
        assert digest(Path(plan["source"]) / relative) == expected, relative
    rows, runs, captures, media = [], [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("train/Train/mean_reward", "Training episode return"),
        ("train/Episode_Metrics/success_rate", "Training success metric"),
        ("train/Policy/mean_std", "Mean normalized action std (all channels)"),
        ("train/Loss/learning_rate", "Adaptive learning rate"),
    ]
    for case in plan["cases"]:
        condition = case["condition"]
        folder = out / condition
        state = read(folder / "state.json")
        training = read(folder / "training.json")
        assert state["status"] == "completed"
        assert (
            state["transitions"]
            == training["final_cumulative_transitions"]
            == plan["transitions_per_policy"]
        )
        assert digest(state["checkpoint"]) == state["checkpoint_sha256"]
        checkpoint = torch.load(state["checkpoint"], map_location="cpu", weights_only=False)
        assert checkpoint["iter"] + 1 == plan["updates"]
        std = checkpoint["actor_state_dict"]["distribution.std_param"].clamp(1e-6, 1e6).numpy()
        history = [
            json.loads(line)
            for line in (Path(state["run_directory"]) / "iterations.jsonl").read_text().splitlines()
        ]
        assert [r["iteration"] for r in history] == list(range(1, plan["updates"] + 1))
        assert all(r["transitions"] == r["iteration"] * plan["num_envs"] * 24 for r in history)
        assert all(
            np.isfinite(v)
            for r in history
            for k, v in r.items()
            if k.startswith("train/") and isinstance(v, (int, float))
        )
        tail = history[-100:]
        runs.append(
            dict(
                condition=condition,
                transitions=state["transitions"],
                checkpoint=state["checkpoint"],
                checkpoint_sha256=state["checkpoint_sha256"],
                completed_at=state["completed_at"],
                final_xyz_std_mm=(std[:3] * 5).tolist(),
                final_all_action_std=std.tolist(),
                last100_mean_return=float(np.mean([r["train/Train/mean_reward"] for r in tail])),
                last100_mean_training_success=float(
                    np.mean([r["train/Episode_Metrics/success_rate"] for r in tail])
                ),
                final_learning_rate=history[-1]["train/Loss/learning_rate"],
            )
        )
        for ax, (key, label) in zip(axes.flat, metrics, strict=True):
            selected = [r for r in history if key in r]
            ax.plot(
                [r["transitions"] / 1e6 for r in selected],
                [r[key] for r in selected],
                label=condition,
                linewidth=1,
            )
            ax.set(xlabel="Million transitions", ylabel=label)
            ax.grid(alpha=0.2)
        for evaluation in state["evaluations"]:
            path = Path(evaluation["path"])
            assert digest(path) == evaluation["sha256"]
            data = read(path)
            assert data["experiment"]["success_state_sample"] == "current_qpos"
            assert data["training_seed"] == 0
            assert (
                data["seed"]
                == plan["evaluation"][
                    "final_seed" if data["split"] == "test" else "validation_seed"
                ]
            )
            assert data["episodes"] == (1024 if data["split"] == "test" else 512)
            assert Path(data["checkpoint"]).name == f"model_{evaluation['updates'] - 1}.pt"
            records = sorted(data["termination_audit"]["records"], key=lambda r: r["episode"])
            assert [r["episode"] for r in records] == list(range(data["episodes"]))
            assert [bool(r["success"]) for r in records] == data["outcomes"]
            assert sum(data["outcomes"]) == data["successes"] == evaluation["successes"]
            assert np.isclose(data["success_rate"], data["successes"] / data["episodes"])
            for key in (
                "success_hold_violations",
                "success_distance_violations",
                "success_three_sample_violations",
                "success_qpos_above_distance_threshold",
            ):
                assert data["termination_audit"][key] == 0, (path, key)
            for r in records:
                assert not r["plug_aligned_history"] or r["plug_raised_history"]
                if r["success"]:
                    assert r["success_hold_steps"] >= 3
                    assert r["qpos_position_error_m"] < 0.002
                    assert all(v < 0.002 for v in r["last_three_manager_errors_m"])
            for variant in ("all", "xm", "xp", "ym", "yp"):
                subset = [
                    r
                    for r in records
                    if variant == "all" or data["variants"][r["episode"]] == variant
                ]
                errors = np.array([r["qpos_position_error_m"] for r in subset]) * 1000
                assert np.isfinite(errors).all()
                successes = sum(r["success"] for r in subset)
                if variant != "all":
                    assert (
                        len(subset)
                        == data["episodes"] // 4
                        == data["per_variant"][variant]["episodes"]
                    )
                    assert successes == data["per_variant"][variant]["successes"]
                failed = [r for r in subset if not r["success"]]
                rows.append(
                    dict(
                        condition=condition,
                        split=data["split"],
                        occlusion=data["experiment"]["occlusion"],
                        transitions=evaluation["transitions"],
                        variant=variant,
                        episodes=len(subset),
                        successes=successes,
                        no_lift=sum(not r["plug_raised_history"] for r in failed),
                        lifted_not_aligned=sum(
                            r["plug_raised_history"] and not r["plug_aligned_history"]
                            for r in failed
                        ),
                        aligned_not_inserted=sum(r["plug_aligned_history"] for r in failed),
                        timeouts=sum(r["timeout"] for r in subset),
                        mean_terminal_error_mm=float(errors.mean()),
                        median_terminal_error_mm=float(np.median(errors)),
                        p90_terminal_error_mm=float(np.quantile(errors, 0.9)),
                        terminal_below_2mm=int((errors < 2).sum()),
                        terminal_below_5mm=int((errors < 5).sum()),
                        source_report=str(path.relative_to(out)),
                        source_sha256=evaluation["sha256"],
                    )
                )
        for setting in ("clean", "dynamic"):
            metadata = read(folder / f"videos-{setting}" / "representatives.json")
            for r in metadata["records"]:
                assert r["expected_success"] == r["reproduced_success"]
                with np.load(r["trace"]) as trace:
                    errors = trace["error"] * 1000
                    captures.append(
                        dict(
                            condition=condition,
                            occlusion=setting,
                            variant=r["variant"],
                            episode=r["episode"],
                            minimum_recorded_error_mm=float(errors.min()),
                            final_recorded_error_mm=float(errors[-1]),
                            trace=str(Path(r["trace"]).relative_to(out)),
                            trace_sha256=digest(r["trace"]),
                        )
                    )
                for key in ("outside_video", "policy_video"):
                    video = Path(r[key])
                    assert video.is_file() and video.stat().st_size > 0
                    media.append(str(video.relative_to(out)))
    for ax in axes.flat:
        ax.legend(fontsize=8)
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "analysis-learning.png", dpi=160)
    plt.close(fig)
    final = [r for r in rows if r["split"] == "test" and r["variant"] == "all"]
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [r["condition"] + " / " + r["occlusion"] for r in final]
    ax.barh(labels, [r["lifted_not_aligned"] for r in final], label="Lifted, never aligned")
    ax.barh(
        labels,
        [r["aligned_not_inserted"] for r in final],
        left=[r["lifted_not_aligned"] for r in final],
        label="Aligned, insertion failed",
    )
    ax.set(xlabel="Failed episodes (1,024 per condition and setting)", xlim=(0, 1024))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    fig.savefig(out / "analysis-stages.png", dpi=160)
    plt.close(fig)
    result = dict(
        built_at=datetime.now(UTC).isoformat(),
        scope="Saved midpoint and final evaluations; no new policy rollouts or training.",
        limitations="One training seed; clean training, dynamic robustness test. Stage flags are history through the previous reward sample. Terminal errors are not trajectory minima. Captured traces are selected representatives, not a random population sample. Exploration shrinkage is descriptive, not causal proof.",
        validation=dict(
            source_files_verified=len(plan["source_sha256"]),
            checkpoints_verified=len(runs),
            evaluations_verified=len(rows) // 5,
            episodes_verified=sum(r["episodes"] for r in rows if r["variant"] == "all"),
            physical_success_violations=0,
            videos_present=len(media),
        ),
        runs=runs,
        evaluations=rows,
        representative_traces=captures,
    )
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "analysis.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results_table = table(
        [
            "Camera",
            "Test",
            "Success",
            "Lifted, not aligned",
            "Aligned, not inserted",
            "Mean / median error (mm)",
        ],
        [
            [
                r["condition"],
                r["occlusion"],
                f"{r['successes']}/{r['episodes']}",
                r["lifted_not_aligned"],
                r["aligned_not_inserted"],
                f"{r['mean_terminal_error_mm']:.2f} / {r['median_terminal_error_mm']:.2f}",
            ]
            for r in final
        ],
    )
    noise_table = table(
        [
            "Camera",
            "Final XYZ request std (mm)",
            "Last 100 updates: mean return",
            "Training success",
            "Final LR",
        ],
        [
            [
                r["condition"],
                ", ".join(f"{v:.4f}" for v in r["final_xyz_std_mm"]),
                f"{r['last100_mean_return']:.2f}",
                f"{r['last100_mean_training_success']:.1%}",
                r["final_learning_rate"],
            ]
            for r in runs
        ],
    )
    links = "".join(f'<li><a href="{html.escape(p)}">{html.escape(p)}</a></li>' for p in media)
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Plug precision: completed-run analysis</title>
<style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#183944}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ccd7df}}img{{max-width:100%}}</style>
<p><a href="index.html">Plug precision report</a></p><h1>Plug precision: completed-run analysis</h1>
<p>All three seed-0 policies completed 18,001,920 transitions. Each scored 0/1,024 on both clean and dynamic final tests; every variant scored 0/256. Midpoint validation also had zero successes. All final episodes reached the lift stage, but alignment was incomplete and no insertion succeeded. These results do not establish a successful camera condition.</p>
<h2>Final physical outcomes</h2>{results_table}<img src="analysis-stages.png" alt="Final failure stages">
<p>Alignment means the episode met the approach-stage condition at least once, not that it maintained alignment. Final object-origin errors and stage history cannot distinguish perception, control, or contact causes.</p>
<h2>Learning and exploration</h2>{noise_table}<img src="analysis-learning.png" alt="Training reward, success, action noise and learning rate">
<p>Initial XYZ request standard deviation was 1.25 mm per axis. Values above are the saved Gaussian standard deviations multiplied by 5 mm, before action clipping and workspace/control limits. Active-camera channels are excluded from the XYZ table but included in the logged mean-std plot. Reduced exploration and positive return without task completion are consistent with learning a partial approach; they do not prove why insertion failed.</p>
<p>The next diagnostic should test whether this exact staged reward and incremental-control recipe is learnable with full state before committing to more RGB training. A successful scripted controller establishes feasibility, but not PPO learnability. No additional jobs were launched by this analysis.</p>
<h2>Evidence</h2><p>{result["validation"]["evaluations_verified"]} evaluations / {result["validation"]["episodes_verified"]:,} episode records checked, {len(runs)} checkpoint hashes and {len(plan["source_sha256"])} frozen source hashes verified. No physical success violations. <a href="analysis.json">Full analysis JSON</a> · <a href="analysis.csv">All variants and milestones CSV</a>.</p>
<p>{html.escape(result["limitations"])}</p><details><summary>Recorded trajectory videos ({len(media)})</summary><ul>{links}</ul></details><p>Built {result["built_at"]}.</p></html>"""
    (out / "analysis.html").write_text(document)
    print(json.dumps(dict(validation=result["validation"], runs=runs, final=final), indent=2))


if __name__ == "__main__":
    main()
