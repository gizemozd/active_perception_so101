"""Build an offline, reproducible plug-training report from existing local evidence.

Run from the repository root with .venv/bin/python scripts/build_plug_report.py.
No training, evaluation, network access, or physics rollout is performed here.
"""

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from active_perception_arms import plug
from active_perception_arms.config import JOINTS, saved_experiment
from active_perception_arms.scenes import native_spec

CONDITIONS = ("wrist", "wrist_static", "initial", "active")
LABELS = {
    "wrist": "Wrist only",
    "wrist_static": "Wrist + fixed view 7",
    "initial": "Initial camera + memory",
    "active": "Active camera + memory",
}
COLORS = dict(zip(CONDITIONS, ("#6b7280", "#2563eb", "#059669", "#d97706"), strict=True))
METRICS = {
    "reward": "train/Train/mean_reward",
    "training_success": "train/Episode_Metrics/success_rate",
    "episode_length": "train/Train/mean_episode_length",
    "learning_rate": "train/Loss/learning_rate",
    "mean_std": "train/Policy/mean_std",
    "value_loss": "train/Loss/value",
    "surrogate_loss": "train/Loss/surrogate",
    "entropy": "train/Loss/entropy",
    "task_reward_rate": "train/Episode_Reward/task",
}


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0, center - half), min(1, center + half)]


def summary(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def smoothed(x, y, width=30):
    if len(y) < width:
        return x, y
    return x[width - 1 :], np.convolve(y, np.ones(width) / width, mode="valid")


def metric_rows(history, key):
    return [(r["transitions"] / 1e6, float(r[key])) for r in history if key in r]


def learning_summary(history):
    rewards = [r for r in history if METRICS["reward"] in r]
    early = [r for r in history[:100] if METRICS["reward"] in r]
    late = rewards[-100:]
    tail = rewards[-300:]
    xs = np.array([r["transitions"] / 1e6 for r in tail])
    ys = np.array([r[METRICS["reward"]] for r in tail])
    slope = float(np.polyfit(xs, ys, 1)[0])
    start = float(np.mean([r[METRICS["reward"]] for r in early]))
    end = float(np.mean([r[METRICS["reward"]] for r in late]))
    fitted_change = slope * float(xs[-1] - xs[0])
    # A descriptive threshold, not a statistical stationarity/convergence test.
    tolerance = max(0.1, abs(float(ys.mean()))) * 0.1
    window_change = float(ys[-150:].mean() - ys[-300:-150].mean())
    return {
        "updates": len(history),
        "contiguous_iterations": [r["iteration"] for r in history]
        == list(range(1, len(history) + 1)),
        "transitions": history[-1]["transitions"],
        "first_100_reward_mean": start,
        "last_100_reward_mean": end,
        "reward_gain": end - start,
        "last_300_reward_std": float(ys.std()),
        "last_300_reward_slope_per_million_transitions": slope,
        "last_300_reward_fitted_change": fitted_change,
        "final_150_vs_previous_150_reward_change": window_change,
        "descriptive_plateau_tolerance": tolerance,
        "descriptive_reward_plateau": max(abs(fitted_change), abs(window_change)) <= tolerance,
        "last_100_lr_at_floor_fraction": float(
            np.mean([r[METRICS["learning_rate"]] <= 1.00001e-5 for r in history[-100:]])
        ),
    }


def configure_plots():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "active_perception_so101",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, assets, name):
    fig.savefig(assets / (name + ".svg"), metadata={"Date": None})
    fig.savefig(assets / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def learning_plots(histories, assets):
    panels = (
        ("reward", "Training episode reward"),
        ("training_success", "Reset-batch success metric"),
        ("episode_length", "Episode length (steps)"),
        ("task_reward_rate", "Native task reward rate"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    for ax, (metric, label) in zip(axes.flat, panels, strict=True):
        for condition, history in histories.items():
            points = metric_rows(history, METRICS[metric])
            x, y = np.asarray(points).T
            ax.plot(x, y, color=COLORS[condition], alpha=0.12, linewidth=0.6)
            sx, sy = smoothed(x, y)
            ax.plot(sx, sy, color=COLORS[condition], label=LABELS[condition], linewidth=1.6)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.axvline(1.2288, color="#999999", alpha=0.5, linestyle="--", linewidth=1)
    axes[0, 1].set_ylim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("Cumulative environment transitions (millions)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Plug insertion · seed 0 · raw traces and 30-update moving means")
    fig.tight_layout()
    save_figure(fig, assets, "learning_curves")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    for ax, (metric, label) in zip(
        axes.flat,
        (
            ("learning_rate", "PPO learning rate"),
            ("mean_std", "Mean action standard deviation"),
            ("value_loss", "Value loss"),
            ("entropy", "Gaussian entropy"),
        ),
        strict=True,
    ):
        for condition, history in histories.items():
            x, y = np.asarray(metric_rows(history, METRICS[metric])).T
            if metric == "learning_rate":
                ax.plot(x, y, color=COLORS[condition], label=LABELS[condition], linewidth=1)
            else:
                sx, sy = smoothed(x, y)
                ax.plot(sx, sy, color=COLORS[condition], label=LABELS[condition], linewidth=1.4)
        if metric == "learning_rate":
            ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Cumulative environment transitions (millions)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Optimization diagnostics · seed 0")
    fig.tight_layout()
    save_figure(fig, assets, "optimization")


def evaluation_plots(reports, checkpoints, assets):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for i, (condition, report) in enumerate(reports.items()):
        p = report["success_rate"] * 100
        low, high = np.asarray(wilson(report["successes"], report["episodes"])) * 100
        axes[0].bar(i, p, color=COLORS[condition])
        axes[0].errorbar(i, p, yerr=[[p - low], [high - p]], color="black", capsize=4)
        axes[0].text(i, high + 1.2, f"{report['successes']}/{report['episodes']}", ha="center")
        for j, variant in enumerate(plug.VARIANTS):
            r = report["per_variant"][variant]
            offset = (i - 1.5) * 0.18
            axes[1].bar(
                j + offset,
                100 * r["success_rate"],
                width=0.18,
                color=COLORS[condition],
                label=LABELS[condition] if j == 0 else None,
            )
    axes[0].set_xticks(range(4), ["Wrist", "Fixed 7", "Initial", "Active"])
    axes[0].set_ylabel("Balanced validation success (%)")
    axes[0].set_ylim(0, 55)
    axes[0].set_title("512 episodes / condition · Wilson 95% intervals")
    axes[1].set_xticks(range(4), plug.VARIANTS)
    axes[1].set_ylabel("Variant validation success (%)")
    axes[1].set_ylim(0, 65)
    axes[1].legend(fontsize=8)
    axes[1].set_title("128 episodes / variant · no xm successes")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, assets, "success_rates")
    fig, ax = plt.subplots(figsize=(9, 4))
    for condition in CONDITIONS:
        points = [r for r in checkpoints if r["condition"] == condition]
        ax.plot(
            [r["transitions"] / 1e6 for r in points],
            [r["success_rate"] * 100 for r in points],
            marker="o",
            label=LABELS[condition],
            color=COLORS[condition],
        )
    ax.set_xlabel("Checkpoint training transitions (millions)")
    ax.set_ylabel("Validation success (%)")
    ax.set_ylim(0, 45)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_title("Checkpoint comparisons · intermediate scores available for wrist / fixed only")
    fig.tight_layout()
    save_figure(fig, assets, "checkpoint_success")


def geometric_visibility(model, data, camera, width, height):
    """Fraction of 18 prong sample points in-frustum and visible by native ray test.

    This is a geometric proxy, not visibility measured in the captured Warp images.
    The sample points lie just inside each prong; a first hit on that prong is visible.
    """
    origin = data.cam_xpos[camera]
    rotation = data.cam_xmat[camera].reshape(3, 3)
    half_fovy = np.deg2rad(model.cam_fovy[camera]) / 2
    tan_y, tan_x = np.tan(half_fovy), np.tan(half_fovy) * width / height
    samples = np.array(
        [
            [0, 0, 0],
            *[[x, y, z] for x in (-0.95, 0.95) for y in (-0.95, 0.95) for z in (-0.95, 0.95)],
        ]
    ) * np.array(plug.PRONG_HALF)
    hits, total = 0, 0
    hit_id = np.empty(1, dtype=np.int32)
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    for name in ("object/prong_a", "object/prong_b"):
        geom = model.geom(name).id
        targets = data.geom_xpos[geom] + samples @ data.geom_xmat[geom].reshape(3, 3).T
        for target in targets:
            total += 1
            direction = target - origin
            local = rotation.T @ direction
            depth = -local[2]
            if depth <= 0 or abs(local[0]) > depth * tan_x or abs(local[1]) > depth * tan_y:
                continue
            direction = direction / np.linalg.norm(direction)
            mujoco.mj_ray(model, data, origin, direction, groups, True, -1, hit_id)
            hits += hit_id[0] == geom
    return float(hits / total)


def analyze_captures(reports, root):
    """Reconstruct pose/contact geometry from stored states, without advancing physics."""
    rows, traces = [], {}
    for condition, report in reports.items():
        cfg = saved_experiment(report["experiment"])
        meta = read(root / report["representative_capture"])
        video_path = (
            root
            / Path(report["representative_capture"]).parent.with_name(
                Path(report["representative_capture"]).parent.name.removesuffix("-traces")
                + "-videos"
            )
            / "representatives.json"
        )
        videos = {r["episode"]: r for r in read(video_path)["records"]}
        for record in meta["records"]:
            episode = record["episode"]
            assert report["outcomes"][episode] == record["expected_success"]
            model = native_spec(replace(cfg, num_envs=1, plug_variant=record["variant"])).compile()
            data = mujoco.MjData(model)
            camera = model.camera("camera_arm/wrist_cam" if cfg.camera_control else "fixed").id
            wrist_camera = model.camera("manipulator/wrist_cam").id
            qindices = np.array([model.joint("camera_arm/" + name).qposadr[0] for name in JOINTS])
            object_adr = model.joint("object/free").qposadr[0]
            with np.load(root / record["trace"]) as arrays:
                positions, rotations, visibility, wrist_visibility, contacts, camera_contacts = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for step in range(len(arrays["qpos"])):
                    data.qpos[:] = arrays["qpos"][step]
                    data.qvel[:] = arrays["qvel"][step]
                    data.qpos[object_adr : object_adr + 3] -= arrays["origins"][step]
                    for body, key in (("fixture/fixture", "fixture"), ("occluder/panel", "panel")):
                        data.mocap_pos[model.body(body).mocapid[0]] = (
                            arrays[key][step] - arrays["origins"][step]
                        )
                    mujoco.mj_forward(model, data)
                    positions.append(data.cam_xpos[camera].copy())
                    rotations.append(data.cam_xmat[camera].reshape(3, 3).copy())
                    visibility.append(
                        geometric_visibility(model, data, camera, cfg.width, cfg.height)
                    )
                    wrist_visibility.append(
                        geometric_visibility(model, data, wrist_camera, cfg.width, cfg.height)
                    )
                    contacts.append(data.ncon)
                    camera_contacts.append(
                        sum(
                            any(
                                model.body(model.geom_bodyid[g]).name.startswith("camera_arm/")
                                for g in contact.geom
                            )
                            for contact in data.contact
                        )
                    )
                positions, rotations = np.asarray(positions), np.asarray(rotations)
                times = np.arange(len(positions)) * cfg.step_dt
                deltas = np.abs(np.diff(arrays["qpos"][:, qindices], axis=0)).sum(-1)
                distances = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
                relative_rotations = rotations[:-1].transpose(0, 2, 1) @ rotations[1:]
                angles = np.arccos(
                    np.clip((np.trace(relative_rotations, axis1=1, axis2=2) - 1) / 2, -1, 1)
                )
                pre = times[:-1] < cfg.initial_seconds
                actions = arrays["action"]
                targets_after = arrays["camera_targets"][times >= cfg.initial_seconds]
                row = {
                    "condition": condition,
                    "variant": record["variant"],
                    "episode": episode,
                    "success": record["expected_success"],
                    "captured_steps": len(times),
                    "last_preterminal_error_mm": float(arrays["error"][-1] * 1000),
                    "minimum_preterminal_error_mm": float(arrays["error"].min() * 1000),
                    "camera_joint_travel_pre_1s_rad": float(deltas[pre].sum()),
                    "camera_joint_travel_post_1s_rad": float(deltas[~pre].sum()),
                    "external_camera_cartesian_travel_m": float(distances.sum()),
                    "external_camera_orientation_travel_rad": float(angles.sum()),
                    "external_geometric_prong_visibility_fraction": float(np.mean(visibility))
                    if "external" in cfg.sensors
                    else None,
                    "external_any_prong_visible_time_fraction": float(
                        np.mean(np.array(visibility) > 0)
                    )
                    if "external" in cfg.sensors
                    else None,
                    "wrist_any_prong_visible_time_fraction": float(
                        np.mean(np.array(wrist_visibility) > 0)
                    ),
                    "native_reconstructed_max_contacts": max(contacts),
                    "native_reconstructed_camera_contact_steps": int(
                        np.count_nonzero(camera_contacts)
                    ),
                    "manipulation_action_clipped_fraction": float(
                        (np.abs(actions[:, :3]) > 1).mean()
                    ),
                    "camera_action_clipped_fraction": float((np.abs(actions[:, 3:]) > 1).mean())
                    if cfg.camera_control
                    else None,
                    "camera_target_max_change_post_1s_rad": float(
                        np.abs(targets_after - targets_after[:1]).max()
                    )
                    if len(targets_after)
                    else None,
                    "trace": record["trace"],
                    "policy_video": videos[episode]["policy_video"],
                    "outside_video": videos[episode]["outside_video"],
                }
                rows.append(row)
                traces[(condition, episode)] = {
                    "time": times,
                    "position": positions,
                    "error": arrays["error"].copy(),
                    "visibility": np.asarray(visibility),
                    "joint_cumulative": np.r_[0, np.cumsum(deltas)],
                }
    return rows, traces


def capture_plots(rows, traces, assets):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, condition in enumerate(CONDITIONS):
        selected = [r for r in rows if r["condition"] == condition]
        pre = np.mean([r["camera_joint_travel_pre_1s_rad"] for r in selected])
        post = np.mean([r["camera_joint_travel_post_1s_rad"] for r in selected])
        axes[0].bar(i, pre, color=COLORS[condition], label="Before 1 s" if i == 0 else None)
        axes[0].bar(
            i,
            post,
            bottom=pre,
            color=COLORS[condition],
            alpha=0.4,
            hatch="//",
            label="After 1 s" if i == 0 else None,
        )
        for row in selected:
            x = i + (0.07 if row["success"] else -0.07)
            axes[1].scatter(
                x,
                row["last_preterminal_error_mm"],
                color=COLORS[condition],
                marker="o" if row["success"] else "x",
                alpha=0.8,
            )
            if row["external_any_prong_visible_time_fraction"] is not None:
                axes[2].scatter(
                    x,
                    100 * row["external_any_prong_visible_time_fraction"],
                    color=COLORS[condition],
                    marker="o" if row["success"] else "x",
                    alpha=0.8,
                )
    axes[0].set_ylabel("Selected-trace mean camera joint travel (rad)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Last captured position error (mm)")
    axes[1].axhline(2, color="black", linewidth=1, linestyle="--")
    axes[2].set_ylabel("Any prong visible: native geometry (% time)")
    for ax in axes:
        ax.set_xticks(range(4), ["Wrist", "Fixed 7", "Initial", "Active"])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Selected first success/failure per variant · diagnostic sample, not population means"
    )
    fig.tight_layout()
    save_figure(fig, assets, "capture_diagnostics")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex="row")
    for i, condition in enumerate(CONDITIONS):
        for row in (r for r in rows if r["condition"] == condition):
            trace = traces[(condition, row["episode"])]
            label = row["variant"] + (" success" if row["success"] else " failure")
            style = "-" if row["success"] else "--"
            axes[0, i].plot(trace["time"], trace["error"] * 1000, style, label=label, linewidth=1)
            axes[1, i].plot(
                trace["time"], trace["joint_cumulative"], style, label=label, linewidth=1
            )
        for ax in axes[:, i]:
            ax.axvline(1, color="#333333", linestyle=":")
            ax.grid(alpha=0.2)
        axes[0, i].axhline(2, color="black", linewidth=0.7)
        axes[0, i].set_title(LABELS[condition], fontsize=10)
        axes[0, i].legend(fontsize=7)
        axes[1, i].set_xlabel("Elapsed time (s)")
    axes[0, 0].set_ylabel("Captured position error (mm)")
    axes[1, 0].set_ylabel("Camera joint travel (rad)")
    fig.tight_layout()
    save_figure(fig, assets, "trace_timecourses")
    fig = plt.figure(figsize=(11, 4.7))
    for i, condition in enumerate(("initial", "active")):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for row in (r for r in rows if r["condition"] == condition):
            position = traces[(condition, row["episode"])]["position"]
            ax.plot(
                *position.T,
                "-" if row["success"] else "--",
                label=row["variant"] + (" success" if row["success"] else " failure"),
            )
            ax.scatter(*position[0], color="black", s=8)
        ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=LABELS[condition])
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, assets, "camera_trajectories")


def collect_reward_repairs(root, output):
    """Keep the fresh screen and resumed stage distinct, with manifest-derived budgets."""
    rows = []
    repair_root = output / "reward_repair"
    for stage, base in (
        ("screen", repair_root),
        ("continuation", repair_root / "continuation"),
        ("exploration", output / "exploration_repair"),
    ):
        for case in ("progress", "legacy_log_hold"):
            directory = base / case
            training_path = directory / "training.json"
            manifest_path = directory / "run_manifest.json"
            training = read(training_path) if training_path.exists() else {}
            manifest = read(manifest_path) if manifest_path.exists() else {}
            initial_path = repair_root / case / "run_manifest.json"
            initial = (
                read(initial_path)
                if stage == "continuation" and initial_path.exists()
                else manifest
            )
            updates = manifest.get("training_updates", manifest.get("total_updates"))
            transitions = training.get(
                "final_cumulative_transitions",
                manifest.get("training_transitions", manifest.get("total_transitions")),
            )
            for repeat in (1, 2):
                path = directory / f"evaluation-repeat{repeat}.json"
                if not path.exists():
                    continue
                value = read(path)
                audit = value.get("termination_audit", {})
                comparison = value.get("repeat_comparison") or {}
                low, high = wilson(value["successes"], value["episodes"])
                row = {
                    "stage": stage,
                    "profile": case,
                    "repeat": repeat,
                    "training_seed": value.get("training_seed", 0),
                    "training_updates": updates,
                    "training_transitions": transitions,
                    "additional_training_transitions": training.get("actual_transitions"),
                    "additional_training_updates": manifest.get("additional_updates"),
                    "training_source_revision": manifest.get("source_revision"),
                    "initial_std": manifest.get("initial_std"),
                    "learning_rate": (manifest.get("optimization") or {}).get("learning_rate"),
                    "lr_schedule": (manifest.get("optimization") or {}).get("schedule"),
                    "entropy_coef": (manifest.get("optimization") or {}).get("entropy_coef"),
                    "initial_training_source_revision": manifest.get(
                        "initial_source_revision", initial.get("source_revision")
                    ),
                    "source_change": manifest.get("source_change"),
                    "source_history": json.dumps(manifest.get("source_history", [])),
                    "training_manifest": str(manifest_path.relative_to(root))
                    if manifest_path.exists()
                    else None,
                    "success_state_sample": value["experiment"].get("success_state_sample"),
                    "reward_profile": value["experiment"].get("reward_profile"),
                    "episodes": value["episodes"],
                    "successes": value["successes"],
                    "success_rate": value["success_rate"],
                    "wilson95_low": low,
                    "wilson95_high": high,
                    "changed_episode_count": comparison.get("changed_episode_count"),
                    "initial_hash_comparable": comparison.get("initial_hash_comparable"),
                    "initial_hash_mismatches": comparison.get("initial_hash_mismatches"),
                    "initial_physics_state_mismatches": comparison.get(
                        "initial_physics_state_mismatches"
                    ),
                    "initial_wrist_image_mismatches": comparison.get(
                        "initial_sensor_image_mismatches", {}
                    ).get("wrist"),
                    "initial_external_image_mismatches": comparison.get(
                        "initial_sensor_image_mismatches", {}
                    ).get("external"),
                    "success_hold_violations": audit.get("success_hold_violations"),
                    "success_distance_violations": audit.get("success_distance_violations"),
                    "success_three_sample_violations": audit.get("success_three_sample_violations"),
                    "success_qpos_above_distance_threshold": audit.get(
                        "success_qpos_above_distance_threshold"
                    ),
                    "report": str(path.relative_to(root)),
                    "wandb_url": value.get("wandb_url"),
                }
                for variant in plug.VARIANTS:
                    detail = value["per_variant"][variant]
                    low, high = wilson(detail["successes"], detail["episodes"])
                    row.update(
                        {
                            variant + "_episodes": detail["episodes"],
                            variant + "_successes": detail["successes"],
                            variant + "_success_rate": detail["success_rate"],
                            variant + "_wilson95_low": low,
                            variant + "_wilson95_high": high,
                        }
                    )
                rows.append(row)
    return rows


def collect_corrected_hparams(root):
    """Read matched old-checkpoint evaluations under the repaired physical criterion."""
    base = root / "artifacts/hparam_search/continuation"
    paths = list(base.glob("*/corrected-evaluation-repeat*.json"))
    if not paths:
        return []
    plan_path = root / "artifacts/hparam_search/next_stage_plan.json"
    plan = read(plan_path)
    cases = {row["case"]: row for row in plan["cases"]}
    audit_manifest = base / "physical-submission.json"
    evaluation_source = (
        read(audit_manifest).get("source_revision") if audit_manifest.exists() else None
    )
    rows = []
    for case in ("baseline", "fixed_lr", "entropy", "both"):
        training_path = base / case / "plug-wrist_static-s0.json"
        training = read(training_path) if training_path.exists() else {}
        transitions = training.get("final_cumulative_transitions", plan["total_budget_per_case"])
        assert transitions == plan["total_budget_per_case"], (
            "Optimizer table requires matched budgets"
        )
        expected = (
            root / plan["baseline_checkpoint"]
            if case == "baseline"
            else Path(cases[case]["expected_final_checkpoint"])
        )
        for repeat in (1, 2):
            path = base / case / f"corrected-evaluation-repeat{repeat}.json"
            if not path.exists():
                continue
            value = read(path)
            checkpoint = Path(value["checkpoint"])
            assert checkpoint.resolve() == expected.resolve()
            match = re.fullmatch(r"model_(\d+)\.pt", checkpoint.name)
            assert match is not None
            updates = int(match[1]) + 1
            assert updates == 751 and value["episodes"] == 512
            assert value["successes"] == sum(value["outcomes"])
            assert value["experiment"]["success_state_sample"] == "current_qpos"
            audit = value.get("termination_audit") or {}
            comparison = value.get("repeat_comparison") or {}
            low, high = wilson(value["successes"], value["episodes"])
            row = {
                "case": case,
                "repeat": repeat,
                "training_seed": value.get("training_seed"),
                "condition": value["experiment"]["condition"],
                "training_updates": updates,
                "training_transitions": transitions,
                "training_budget_source": str(
                    (training_path if training_path.exists() else plan_path).relative_to(root)
                ),
                "evaluation_source_revision": evaluation_source,
                "success_state_sample": value["experiment"]["success_state_sample"],
                "split": value["split"],
                "evaluation_seed_start": value["seed"],
                "evaluation_num_envs": value["experiment"]["num_envs"],
                "episodes": value["episodes"],
                "successes": value["successes"],
                "success_rate": value["success_rate"],
                "wilson95_low": low,
                "wilson95_high": high,
                "changed_episode_count": comparison.get("changed_episode_count"),
                "initial_hash_comparable": comparison.get("initial_hash_comparable"),
                "initial_hash_mismatches": comparison.get("initial_hash_mismatches"),
                "initial_physics_state_mismatches": comparison.get(
                    "initial_physics_state_mismatches"
                ),
                "initial_wrist_image_mismatches": comparison.get(
                    "initial_sensor_image_mismatches", {}
                ).get("wrist"),
                "initial_external_image_mismatches": comparison.get(
                    "initial_sensor_image_mismatches", {}
                ).get("external"),
                "success_hold_violations": audit.get("success_hold_violations"),
                "success_distance_violations": audit.get("success_distance_violations"),
                "success_three_sample_violations": audit.get("success_three_sample_violations"),
                "success_qpos_above_distance_threshold": audit.get(
                    "success_qpos_above_distance_threshold"
                ),
                "report": str(path.relative_to(root)),
                "checkpoint": str(checkpoint),
                "wandb_url": value.get("wandb_url"),
            }
            for variant in plug.VARIANTS:
                detail = value["per_variant"][variant]
                assert detail["episodes"] == 128
                low, high = wilson(detail["successes"], detail["episodes"])
                row.update(
                    {
                        variant + "_episodes": detail["episodes"],
                        variant + "_successes": detail["successes"],
                        variant + "_success_rate": detail["success_rate"],
                        variant + "_wilson95_low": low,
                        variant + "_wilson95_high": high,
                    }
                )
            rows.append(row)
    return rows


def write_corrected_hparam_tables(
    output,
    rows,
    stem="corrected_hparam_metrics",
    scope=None,
    label_key="case",
    label_title="Optimizer",
):
    """Export physical scores without pooling numerical repeats as independent episodes."""
    if not rows:
        return
    write_csv(output / (stem + ".csv"), rows)
    latex = [
        scope
        or "% Corrected current_qpos evaluations of OLD-TRAINED model_750 checkpoints; no retraining or optimizer updates in this audit.",
        "% Three discrete consecutive physical-state samples at 25 Hz, not continuous dwell. Each checkpoint: 751 updates / 9,228,288 training transitions; seed 0 only.",
        "% Each row is a separate 512-episode balanced validation execution. Do not pool repeats as 1,024 independent samples or training seeds. Wilson intervals describe episode uncertainty only.",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline",
        label_title
        + r" & Repeat & Success / $N$ & Success (\%) & 95\% CI & $x-$ & $x+$ & $y-$ & $y+$ \\",
        r"\hline",
    ]
    for row in rows:
        latex.append(
            row[label_key].replace("_", r"\_")
            + f" & {row['repeat']} & {row['successes']}/{row['episodes']} & "
            + f"{100 * row['success_rate']:.2f} & [{100 * row['wilson95_low']:.2f}, {100 * row['wilson95_high']:.2f}] & "
            + " & ".join(f"{100 * row[v + '_success_rate']:.2f}" for v in plug.VARIANTS)
            + r" \\"
        )
    latex += [
        r"\hline",
        r"\end{tabular}",
        f"% Variant columns are percentages; per-variant Wilson intervals and termination/repeat audits are in {stem}.csv.",
    ]
    (output / (stem + ".tex")).write_text("\n".join(latex) + "\n")


def collect_repair_learning(root, output, family="reward_repair", fresh=False):
    """Use frozen screen budgets; expose all completed continuation updates when available."""
    rows, summaries, histories = [], [], {}
    for case in ("progress", "legacy_log_hold"):
        screen_path = output / family / case / "run_manifest.json"
        if not screen_path.exists():
            continue
        screen = read(screen_path)
        screen_updates = screen["training_updates"]
        continuation_path = output / family / "continuation" / case / "run_manifest.json"
        continuation = read(continuation_path) if not fresh and continuation_path.exists() else {}
        training_path = (screen_path if fresh else continuation_path).parent / "training.json"
        training = read(training_path) if training_path.exists() else {}
        if not screen.get("run_directory"):
            continue
        history_path = Path(screen["run_directory"]) / "iterations.jsonl"
        if not history_path.exists():
            continue
        lines = history_path.read_text().splitlines()
        expected_updates = continuation.get("training_updates", screen_updates)
        expected_transitions = continuation.get(
            "training_transitions", screen["training_transitions"]
        )
        completed = (
            (fresh or bool(continuation))
            and training.get("final_cumulative_transitions") == expected_transitions
            and len(lines) >= expected_updates
        )
        if fresh and not completed:
            continue
        selected_updates = expected_updates if completed else screen_updates
        # The mutable runner file is never consulted. Initial rows retain their original budget/source.
        history = [json.loads(line) for line in lines[:selected_updates]]
        assert [r["iteration"] for r in history] == list(range(1, selected_updates + 1))
        assert history[-1]["transitions"] == (
            expected_transitions if completed else screen["training_transitions"]
        )
        stage = "exploration" if fresh else "continuation" if completed else "screen"
        histories[case] = history
        for row in history:
            rows.append(
                {
                    "profile": case,
                    "plotted_stage": stage,
                    "training_source_revision": screen["source_revision"]
                    if row["iteration"] <= screen_updates
                    else continuation["source_revision"],
                    **row,
                }
            )
        rewards = [r for r in history if METRICS["reward"] in r]
        comparison_window = min(100, len(history) // 2)
        first = np.array(
            [r[METRICS["reward"]] for r in history[:comparison_window] if METRICS["reward"] in r]
        )
        last = np.array(
            [r[METRICS["reward"]] for r in history[-comparison_window:] if METRICS["reward"] in r]
        )
        summary = {
            "profile": case,
            "plotted_stage": stage,
            "screen_updates": screen_updates,
            "plotted_updates": len(history),
            "available_history_lines": len(lines),
            "expected_total_updates": expected_updates,
            "plotted_transitions": history[-1]["transitions"],
            "expected_total_transitions": expected_transitions,
            "continuation_training_complete": completed,
            "training_complete": completed,
            "reward_comparison_window_updates": comparison_window,
            "first_window_reward_samples": len(first),
            "last_window_reward_samples": len(last),
            "first_window_reward_mean": float(first.mean()) if first.size else None,
            "last_window_reward_mean": float(last.mean()) if last.size else None,
            "reward_gain": float(last.mean() - first.mean()) if first.size and last.size else None,
            "tail_300_available": len(rewards) >= 300,
            "last_300_reward_slope_per_million_transitions": None,
            "last_300_reward_fitted_change": None,
            "final_150_vs_previous_150_reward_change": None,
            "descriptive_plateau_tolerance": None,
            "descriptive_reward_plateau": None,
            "manifest": str(
                (continuation_path if completed and not fresh else screen_path).relative_to(root)
            ),
            "history": str(history_path.relative_to(root)),
        }
        if len(rewards) >= 300:
            tail = rewards[-300:]
            x = np.array([r["transitions"] / 1e6 for r in tail])
            y = np.array([r[METRICS["reward"]] for r in tail])
            slope = float(np.polyfit(x, y, 1)[0])
            fitted_change = slope * float(x[-1] - x[0])
            window_change = float(y[-150:].mean() - y[:150].mean())
            tolerance = 0.1 * max(0.1, abs(float(y.mean())))
            summary.update(
                {
                    "last_300_reward_slope_per_million_transitions": slope,
                    "last_300_reward_fitted_change": fitted_change,
                    "final_150_vs_previous_150_reward_change": window_change,
                    "descriptive_plateau_tolerance": tolerance,
                    "descriptive_reward_plateau": max(abs(fitted_change), abs(window_change))
                    <= tolerance,
                }
            )
        tail = history[-min(300, len(history)) :]
        episodes = sum(r.get("train/Loss/diagnostic_completed_episodes", 0) for r in tail)
        successes = sum(r.get("train/Loss/diagnostic_successes", 0) for r in tail)
        summary["tail_completed_episodes"] = episodes
        summary["tail_episode_weighted_training_success"] = (
            successes / episodes if episodes else None
        )
        kls = [r["train/Loss/diagnostic_kl"] for r in tail if "train/Loss/diagnostic_kl" in r]
        summary["tail_mean_diagnostic_kl"] = float(np.mean(kls)) if kls else None
        summary["last_mean_action_std"] = history[-1].get(METRICS["mean_std"])
        summaries.append(summary)
    return rows, summaries, histories


def repair_learning_plots(
    histories,
    assets,
    screen_budgets,
    asset_prefix="repair_learning_",
    title="Corrected-training reward profile",
):
    """Keep the two objective scales on distinct figures; success smoothing weights episodes."""
    for case, history in histories.items():
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        panels = (
            (METRICS["reward"], "Training episode reward · this profile's scale"),
            ("train/Loss/diagnostic_episode_weighted_success", "Episode-weighted training success"),
            ("train/Loss/diagnostic_kl", "PPO diagnostic KL"),
            (METRICS["mean_std"], "Action standard deviation"),
            (METRICS["learning_rate"], "PPO learning rate"),
            ("train/Loss/diagnostic_clip_fraction", "PPO clipped fraction"),
        )
        for ax, (key, label) in zip(axes.flat, panels, strict=True):
            points = metric_rows(history, key)
            if points:
                x, y = np.asarray(points).T
                ax.plot(x, y, alpha=0.2, linewidth=0.7, color="#2563eb")
                if key == "train/Loss/diagnostic_episode_weighted_success":
                    available = [r for r in history if key in r]
                    width = min(30, len(available))
                    completed = np.array(
                        [r["train/Loss/diagnostic_completed_episodes"] for r in available]
                    )
                    successful = np.array([r["train/Loss/diagnostic_successes"] for r in available])
                    denominator = np.convolve(completed, np.ones(width), mode="valid")
                    numerator = np.convolve(successful, np.ones(width), mode="valid")
                    weighted = np.divide(
                        numerator,
                        denominator,
                        out=np.full_like(numerator, np.nan),
                        where=denominator > 0,
                    )
                    ax.plot(x[width - 1 :], weighted, color="#2563eb", linewidth=1.6)
                    ax.set_ylim(0, 1)
                elif key == METRICS["mean_std"]:
                    std_keys = sorted(
                        {
                            k
                            for r in history
                            for k in r
                            if re.fullmatch(r"train/Loss/diagnostic_action_std_\d+", k)
                        }
                    )
                    for std_key in std_keys:
                        std_points = metric_rows(history, std_key)
                        sx, sy = np.asarray(std_points).T
                        ax.plot(sx, sy, linewidth=1.2, label="action " + std_key.rsplit("_", 1)[-1])
                    ax.legend(fontsize=8)
                else:
                    sx, sy = smoothed(x, y)
                    ax.plot(sx, sy, color="#2563eb", linewidth=1.6)
                if key == METRICS["learning_rate"]:
                    ax.set_yscale("log")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            # This marks the immutable screen budget, not the mutable runner max_iterations.
            screen_updates = screen_budgets[case]
            if len(history) > screen_updates:
                ax.axvline(
                    history[screen_updates - 1]["transitions"] / 1e6,
                    color="#777",
                    linestyle="--",
                    linewidth=1,
                )
        for ax in axes[-1]:
            ax.set_xlabel("Cumulative environment transitions (millions)")
        fig.suptitle(f"{title}: {case} · seed 0 · {len(history)} recorded updates")
        fig.tight_layout()
        save_figure(fig, assets, asset_prefix + case)


def reward_video_cards(root, output, directory, case, stage_label):
    """Show one captured example per available outcome, without implying variant coverage."""
    metadata = directory / "evaluation-repeat1-videos/representatives.json"
    if not metadata.exists():
        return []
    records = read(metadata)["records"]
    selected = [
        next((r for r in records if bool(r["expected_success"]) == success), None)
        for success in (True, False)
    ]
    cards = []
    for record in selected:
        if not record or not all(
            (root / record[key]).exists() for key in ("outside_video", "policy_video")
        ):
            continue
        cards.append(
            f'<article class="video-card"><h3>{html.escape(case)} · {html.escape(stage_label)} · '
            f"{html.escape(record['variant'])} · {'success' if record['expected_success'] else 'failure'} · "
            f'episode {record["episode"]}</h3><p class="small">First captured example of this outcome '
            "in repeat 1, when present; actual scored trajectory and actor inputs. "
            'These selected examples do not represent every variant.</p><div class="video-pair">'
        )
        for key in ("outside_video", "policy_video"):
            src = html.escape(os.path.relpath(root / record[key], output))
            cards.append(f'<video controls preload="none" playsinline src="{src}"></video>')
        cards.append(
            '</div><button type="button" onclick="playPair(this)">Play pair</button></article>'
        )
    return cards


def collect_exploration_status(root, output):
    """Report recorded reservations/runtime without inferring live Slurm state or results."""
    rows = []
    for case in ("progress", "legacy_log_hold"):
        directory = output / "exploration_repair" / case
        path = directory / "run_manifest.json"
        if not path.exists():
            continue
        manifest = read(path)
        runtime_path = directory / "training-runtime.json"
        runtime = read(runtime_path) if runtime_path.exists() else {}
        evaluation_path = directory / "evaluation-submission.json"
        evaluation = read(evaluation_path) if evaluation_path.exists() else {}
        rows.append(
            {
                "profile": case,
                "recorded_manifest_status": manifest.get("status"),
                "recorded_training_runtime_status": runtime.get("status"),
                "training_job_id": manifest.get("training_job_id", manifest.get("job_id")),
                "evaluation_job_id": manifest.get("evaluation_job_id", evaluation.get("job_id")),
                "target_updates": manifest.get("training_updates"),
                "target_transitions": manifest.get("training_transitions"),
                "source_revision": manifest.get("source_revision"),
                "manifest": str(path.relative_to(root)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("artifacts/plug_analysis"))
    parser.add_argument(
        "--validate-media", action="store_true", help="Decode first/middle/last frames"
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Build a portable HTML/media ZIP without checkpoints or raw NPZ traces",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    configure_plots()
    manifests = {
        r["condition"]: r for r in read(root / "artifacts/cluster_continuation/manifest.json")
    }
    reports, histories, convergence = {}, {}, {}
    checkpoints, metrics, episodes, learning_rows = [], [], [], []
    for condition in CONDITIONS:
        manifest = manifests[condition]
        report_path = root / manifest["evaluation_output"]
        reports[condition] = report = read(report_path)
        history_path = root / manifest["directory"] / "iterations.jsonl"
        histories[condition] = history = [
            json.loads(line) for line in history_path.read_text().splitlines()
        ]
        convergence[condition] = learning_summary(history)
        assert report["successes"] == sum(report["outcomes"])
        assert report["episodes"] == len(report["outcomes"]) == 512
        assert all(r["episodes"] == 128 for r in report["per_variant"].values())
        assert convergence[condition]["contiguous_iterations"]
        assert history[-1]["transitions"] == 18432000
        training = read(root / manifest["training_result"])
        low, high = wilson(report["successes"], report["episodes"])
        row = {
            "condition": condition,
            "condition_label": LABELS[condition],
            "training_seed": 0,
            "training_seeds_count": 1,
            "training_transitions": 18432000,
            "checkpoint": report["checkpoint"],
            "split": report["split"],
            "evaluation_seed_start": report["seed"],
            "evaluation_num_envs": 128,
            "episodes": report["episodes"],
            "successes": report["successes"],
            "success_rate": report["success_rate"],
            "wilson95_low": low,
            "wilson95_high": high,
            "mean_episode_seconds": report["mean_episode_seconds"],
            "mean_success_completion_seconds": report["mean_success_completion_seconds"],
            "unsuccessful_fraction": 1 - report["success_rate"],
            "timeout_rate": None,
            "failure_termination_rate": None,
            "mean_camera_joint_travel_rad": report["mean_camera_joint_travel_rad"],
            "training_gpu_hours": training["cumulative_gpu_hours_including_wandb_finish"],
            "training_wandb_url": manifest["wandb"]["url"],
            "evaluation_wandb_url": report["wandb_url"],
            "source_revision": manifest["source_revision"],
            "occlusion": "clean",
            "success_state_sample": report["experiment"].get(
                "success_state_sample", "derived_substep"
            ),
            "memory": "gru",
            "fixed_view_index": 7 if condition == "wrist_static" else None,
            "evaluation_report": str(report_path.relative_to(root)),
        }
        for variant in plug.VARIANTS:
            r = report["per_variant"][variant]
            lo, hi = wilson(r["successes"], r["episodes"])
            row.update(
                {
                    variant + "_episodes": r["episodes"],
                    variant + "_successes": r["successes"],
                    variant + "_success_rate": r["success_rate"],
                    variant + "_wilson95_low": lo,
                    variant + "_wilson95_high": hi,
                }
            )
        metrics.append(row)
        for i, (won, time, travel, variant) in enumerate(
            zip(
                report["outcomes"],
                report["episode_seconds"],
                report["camera_joint_travel_rad"],
                report["variants"],
                strict=True,
            )
        ):
            episodes.append(
                {
                    "condition": condition,
                    "training_seed": 0,
                    "episode": i,
                    "reset_seed": report["seed"] + i // 128,
                    "variant": variant,
                    "success": won,
                    "episode_seconds": time,
                    "camera_joint_travel_rad": travel,
                }
            )
        for r in history:
            learning_rows.append(
                {
                    "condition": condition,
                    "seed": 0,
                    "iteration": r["iteration"],
                    "transitions": r["transitions"],
                    **{name: r.get(key) for name, key in METRICS.items()},
                }
            )
        for transitions, path in (
            (1228800, root / f"artifacts/cluster_pilot/evaluation-{condition}-s0.json"),
            (
                9228288,
                root
                / f"artifacts/cluster_continuation/interim-evaluation-{condition}-s0-i750.json",
            ),
            (18432000, report_path),
        ):
            if path.exists():
                r = read(path)
                checkpoints.append(
                    {
                        "condition": condition,
                        "transitions": transitions,
                        "successes": r["successes"],
                        "episodes": r["episodes"],
                        "success_rate": r["success_rate"],
                        "report": str(path.relative_to(root)),
                    }
                )
    write_csv(output / "performance_metrics.csv", metrics)
    write_csv(output / "evaluation_episodes.csv", episodes)
    write_csv(output / "learning_curves.csv", learning_rows)
    write_csv(output / "checkpoint_metrics.csv", checkpoints)
    learning_plots(histories, assets)
    evaluation_plots(reports, checkpoints, assets)
    capture_rows, traces = analyze_captures(reports, root)
    write_csv(output / "representative_trajectories.csv", capture_rows)
    capture_plots(capture_rows, traces, assets)
    comparisons = [
        {
            "baseline": condition,
            "active_minus_baseline_percentage_points": 100
            * (reports["active"]["success_rate"] - report["success_rate"]),
        }
        for condition, report in reports.items()
        if condition != "active"
    ]
    tuning = []
    for path in sorted((root / "artifacts/hparam_search/continuation").glob("**/evaluation*.json")):
        value = read(path)
        if "success_rate" in value:
            tuning.append(
                {
                    "case": path.parent.name,
                    "successes": value["successes"],
                    "episodes": value["episodes"],
                    "success_rate": value["success_rate"],
                    "report": str(path.relative_to(root)),
                    "wandb_url": value.get("wandb_url"),
                }
            )
    repair_rows = collect_reward_repairs(root, output)
    corrected_hparams = collect_corrected_hparams(root)
    write_corrected_hparam_tables(output, corrected_hparams)
    repair_learning_rows, repair_learning_summaries, repair_histories = collect_repair_learning(
        root, output
    )
    write_csv(output / "repair_learning_curves.csv", repair_learning_rows)
    write_json(output / "repair_learning_summary.json", repair_learning_summaries)
    repair_learning_plots(
        repair_histories,
        assets,
        {r["profile"]: r["screen_updates"] for r in repair_learning_summaries},
    )
    reward_repair = [row for row in repair_rows if row["stage"] == "screen"]
    reward_continuation = [row for row in repair_rows if row["stage"] == "continuation"]
    exploration_repair = [row for row in repair_rows if row["stage"] == "exploration"]
    exploration_status = collect_exploration_status(root, output)
    write_csv(output / "reward_repair_metrics.csv", reward_repair)
    write_csv(output / "reward_repair_continuation_metrics.csv", reward_continuation)
    write_corrected_hparam_tables(
        output,
        exploration_repair,
        stem="exploration_repair_metrics",
        scope="% Fresh exploration-configuration training with current_qpos criterion and critic-bootstrap correction from update 1. Initial std and entropy change jointly; not an isolated std ablation or an exact historical reproduction. Separate from reward screen and resumed continuations.",
        label_key="profile",
        label_title="Reward profile",
    )
    exploration_learning_rows, exploration_learning, exploration_histories = (
        collect_repair_learning(root, output, family="exploration_repair", fresh=True)
    )
    write_csv(output / "exploration_learning_curves.csv", exploration_learning_rows)
    write_json(output / "exploration_learning_summary.json", exploration_learning)
    repair_learning_plots(
        exploration_histories,
        assets,
        {r["profile"]: r["screen_updates"] for r in exploration_learning},
        asset_prefix="exploration_learning_",
        title="Fresh joint exploration configuration",
    )
    termination_audits = []
    audit_paths = sorted(
        set(
            path
            for pattern in ("audit-*.json", "termination-*.json", "corrected-*-s0.json")
            for path in output.glob(pattern)
        )
    )
    for path in audit_paths:
        value = read(path)
        if "termination_audit" not in value:
            continue
        termination_audits.append(
            {
                "report": str(path.relative_to(root)),
                "condition": value["experiment"]["condition"],
                "success_state_sample": value["experiment"].get(
                    "success_state_sample", "derived_substep"
                ),
                "episodes": value["episodes"],
                "successes": value["successes"],
                "success_rate": value["success_rate"],
                "termination_audit": {
                    k: v for k, v in value["termination_audit"].items() if k != "records"
                },
                "repeat_comparison": value.get("repeat_comparison"),
            }
        )
    legacy_path = output / "legacy_run_inventory.json"
    legacy = read(legacy_path) if legacy_path.exists() else None
    corrected_metrics, corrected_captures, corrected_reports = [], [], {}
    for path in sorted(output.glob("corrected-*-s0.json")):
        report = read(path)
        condition = report["experiment"]["condition"]
        row = next(r for r in metrics if r["condition"] == condition).copy()
        low, high = wilson(report["successes"], report["episodes"])
        audit = report.get("termination_audit", {})
        flags = audit.get("terminal_flags", {})
        row.update(
            {
                "checkpoint": report["checkpoint"],
                "episodes": report["episodes"],
                "successes": report["successes"],
                "success_rate": report["success_rate"],
                "wilson95_low": low,
                "wilson95_high": high,
                "success_state_sample": report["experiment"]["success_state_sample"],
                "mean_episode_seconds": report["mean_episode_seconds"],
                "mean_success_completion_seconds": report["mean_success_completion_seconds"],
                "mean_camera_joint_travel_rad": report["mean_camera_joint_travel_rad"],
                "unsuccessful_fraction": 1 - report["success_rate"],
                "timeout_flag_fraction": flags.get("timeout", 0) / report["episodes"]
                if flags
                else None,
                "failure_termination_rate": flags.get("failure", 0) / report["episodes"]
                if flags
                else None,
                "mean_terminal_position_error_m": audit.get("mean_terminal_position_error_m"),
                "evaluation_report": str(path.relative_to(root)),
                "evaluation_wandb_url": report.get("wandb_url"),
            }
        )
        for variant in plug.VARIANTS:
            r = report["per_variant"][variant]
            low, high = wilson(r["successes"], r["episodes"])
            row.update(
                {
                    variant + "_episodes": r["episodes"],
                    variant + "_successes": r["successes"],
                    variant + "_success_rate": r["success_rate"],
                    variant + "_wilson95_low": low,
                    variant + "_wilson95_high": high,
                }
            )
        corrected_metrics.append(row)
        if report.get("representative_capture"):
            capture_path = root / report["representative_capture"]
            video_meta = (
                capture_path.parent.with_name(
                    capture_path.parent.name.removesuffix("-traces") + "-videos"
                )
                / "representatives.json"
            )
            if video_meta.exists() and all(
                (root / r[k]).exists()
                for r in read(video_meta)["records"]
                for k in ("outside_video", "policy_video")
            ):
                corrected_reports[condition] = report
    if corrected_reports:
        corrected_captures, _ = analyze_captures(corrected_reports, root)
        for row in corrected_captures:
            row["poster_prefix"] = "corrected-"
        write_csv(output / "corrected_representative_trajectories.csv", corrected_captures)
    write_csv(output / "corrected_performance_metrics.csv", corrected_metrics)
    analysis = {
        "generated_at": datetime.now(UTC).isoformat(),
        "training_gate": "NOT_RELIABLE_ACROSS_VARIANTS",
        "conclusion": "Initial-only camera movement with fresh images is promising in seed 0; continual active motion has no demonstrated benefit over initial positioning plus memory.",
        "evaluation_protocol": {
            "split": "validation",
            "training_seeds": [0],
            "episodes_per_condition": 512,
            "episodes_per_variant": 128,
            "reset_seeds": [10000, 10001, 10002, 10003],
            "training_transitions": 18432000,
        },
        "convergence": convergence,
        "performance_metrics": metrics,
        "comparisons": comparisons,
        "representative_count": len(capture_rows),
        "tuning_continuation_results": tuning,
        "corrected_hparam_results": corrected_hparams,
        "reward_repair_learning": repair_learning_summaries,
        "reward_repair_results": reward_repair,
        "reward_repair_continuation_results": reward_continuation,
        "exploration_repair_results": exploration_repair,
        "exploration_repair_learning": exploration_learning,
        "exploration_repair_status": exploration_status,
        "headline_success_state_sample": "current_qpos"
        if len(corrected_metrics) == 4
        else "derived_substep",
        "corrected_comparisons": [
            {
                "baseline": r["condition"],
                "success_state_sample": "current_qpos",
                "active_minus_baseline_percentage_points": 100
                * (active["success_rate"] - r["success_rate"]),
            }
            for r in corrected_metrics
            if r["condition"] != "active"
            for active in corrected_metrics
            if active["condition"] == "active"
        ],
        "termination_audits": termination_audits,
        "corrected_performance_metrics": corrected_metrics,
        "corrected_representative_count": len(corrected_captures),
        "historical_run_inventory": "artifacts/plug_analysis/legacy_run_inventory.json"
        if legacy
        else None,
        "limitations": [
            "One training seed; Wilson intervals cover episode uncertainty only.",
            "Validation used for diagnosis, not an untouched test split.",
            "Fixed view 7 was not selected by a completed validation-success search.",
            "Training success is reset-batch averaged, not episode weighted.",
            "Representative captures are outcome-selected; no population inference.",
            "Geometric visibility/contact reconstruction uses native MuJoCo, not recorded Warp segmentation or contact telemetry.",
            "Pre-step captures omit the final terminal state; final errors are last preterminal samples.",
            "Original evaluations do not separate timeouts from failure terminations.",
            "Paired seed labels do not establish deterministic episode-level replay.",
            "Final seeded repeats disagree on 86/512 initial-only and 125/512 active episode outcomes (16.8% and 24.4%), despite identical initial state/image hashes.",
            "Three consecutive 25 Hz success samples span 0.08 s between first/last sample; they do not prove a continuous 0.12 s dwell.",
        ],
    }
    write_json(output / "analysis.json", analysis)
    latex = [
        r"% HISTORICAL derived_substep scoring: lagged manager geometry, not current physical qpos. Audit found physical threshold misses and substantial repeat disagreement; do not treat this as corrected insertion success.",
        r"% Wilson intervals represent episode sampling uncertainty, not training-seed variance.",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Camera condition & Legacy score (\%) & 95\% CI & $x-$ & $x+$ & $y-$ & $y+$ \\",
        r"\hline",
    ]
    for row in metrics:
        latex.append(
            f"{LABELS[row['condition']]} & {100 * row['success_rate']:.2f} & "
            f"[{100 * row['wilson95_low']:.2f}, {100 * row['wilson95_high']:.2f}] & "
            + " & ".join(f"{100 * row[v + '_success_rate']:.2f}" for v in plug.VARIANTS)
            + r" \\"
        )
    latex += [
        r"\hline",
        r"\end{tabular}",
        "% Each condition: 18,432,000 transitions, one training seed, 512 balanced validation episodes.",
    ]
    (output / "performance_metrics.tex").write_text("\n".join(latex) + "\n")
    if corrected_metrics:
        corrected_latex = (
            latex[:6]
            + [
                f"{LABELS[r['condition']]} & {100 * r['success_rate']:.2f} & "
                f"[{100 * r['wilson95_low']:.2f}, {100 * r['wilson95_high']:.2f}] & "
                + " & ".join(f"{100 * r[v + '_success_rate']:.2f}" for v in plug.VARIANTS)
                + r" \\"
                for r in corrected_metrics
            ]
            + latex[-3:]
        )
        corrected_latex[0] = (
            "% Corrected current-qpos criterion: three discrete consecutive samples at 25 Hz; not continuous dwell. Exploratory seed-0 validation."
        )
        corrected_latex[4] = (
            r"Camera condition & Success (\%) & 95\% CI & $x-$ & $x+$ & $y-$ & $y+$ \\"
        )
        (output / "corrected_performance_metrics.tex").write_text("\n".join(corrected_latex) + "\n")
    create_video_posters(root, assets, capture_rows + corrected_captures)
    build_html(output, root, analysis, capture_rows, tuning, legacy, corrected_captures)
    validation = validate_report(
        output, root, capture_rows + corrected_captures, args.validate_media
    )
    write_json(output / "report_validation.json", validation)
    if args.bundle:
        bundle_report(root, output)
    print(
        json.dumps(
            {
                "report": str(output / "index.html"),
                "legacy_successes": {c: r["successes"] for c, r in reports.items()},
                "corrected_successes": {c: r["successes"] for c, r in corrected_reports.items()},
                "representative_videos": 2 * (len(capture_rows) + len(corrected_captures)),
                "validation": {k: v for k, v in validation.items() if k != "videos"},
            },
            indent=2,
        )
    )


def create_video_posters(root, assets, rows):
    import imageio.v2 as imageio
    from PIL import Image

    for row in rows:
        for key in ("outside_video", "policy_video"):
            target = (
                assets
                / f"{row.get('poster_prefix', '')}{row['condition']}_{row['episode']}_{key}.jpg"
            )
            if target.exists():
                continue
            reader = imageio.get_reader(root / row[key])
            try:
                frame = Image.fromarray(reader.get_data(max(0, row["captured_steps"] - 2)))
                frame.thumbnail((768, 432))
                frame.save(target, quality=85)
            finally:
                reader.close()


def build_html(output, root, analysis, captures, tuning, legacy, corrected_captures):
    esc = html.escape

    def link(path, title):
        target = Path(path)
        if not target.is_absolute():
            target = root / target
        return f'<a href="{esc(os.path.relpath(target, output))}">{esc(title)}</a>'

    def figure(name, caption):
        return (
            f'<figure><img src="assets/{name}.svg" alt="{esc(caption)}"><figcaption>{esc(caption)} '
        )
        # Caption close and PDF link are added by the caller below.

    def plot(name, caption):
        return figure(name, caption) + f'<a href="assets/{name}.pdf">PDF</a></figcaption></figure>'

    def table(headers, rows):
        return (
            '<div class="table-wrap"><table><thead><tr>'
            + "".join(f"<th>{esc(h)}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
            + "</tbody></table></div>"
        )

    metrics = analysis["performance_metrics"]
    corrected_complete = len(analysis["corrected_performance_metrics"]) == len(CONDITIONS)
    headline_metrics = analysis["corrected_performance_metrics"] if corrected_complete else metrics
    headline_metrics = sorted(headline_metrics, key=lambda r: CONDITIONS.index(r["condition"]))
    headline = {r["condition"]: r for r in headline_metrics}
    criterion = (
        "corrected current-qpos criterion"
        if corrected_complete
        else "original derived-substep scoring"
    )
    sections = [
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Plug insertion · active perception report</title>
<style>
:root { color-scheme:light; font-family:system-ui,-apple-system,sans-serif; color:#1f2937; background:#f4f6f8; }
body { margin:0; } main { max-width:1280px; margin:auto; padding:36px 28px 90px; }
h1 { font-size:2.2rem; margin:8px 0 16px; } h2 { margin:48px 0 18px; font-size:1.55rem; }
h3 { margin:28px 0 10px; } p,li { line-height:1.65; } a { color:#155bbb; } code { font-size:.9em; }
.eyebrow { color:#536477; font-size:.85rem; letter-spacing:.08em; text-transform:uppercase; }
.notice { background:#fff7e5; border-left:4px solid #d97706; padding:14px 20px; border-radius:4px; }
.cards { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:25px 0; }
.card,figure,.video-card { background:white; border:1px solid #dce2e8; border-radius:10px; }
.card { padding:18px; } .card strong { display:block; font-size:2rem; margin:8px 0; }
.small,figcaption { color:#556578; font-size:.88rem; line-height:1.5; }
figure { margin:24px 0; padding:14px; } figure img { width:100%; height:auto; } figcaption { margin:10px 8px 4px; }
.table-wrap { overflow-x:auto; margin:20px 0; } table { border-collapse:collapse; width:100%; background:white; }
th,td { padding:11px 13px; border-bottom:1px solid #e2e7ed; text-align:left; white-space:nowrap; }
th { font-size:.83rem; color:#526173; background:#edf1f5; } tbody tr:hover { background:#f8fafc; }
nav { display:flex; gap:18px; flex-wrap:wrap; margin:26px 0; } .downloads { display:flex; gap:20px; flex-wrap:wrap; }
details { margin:14px 0; } summary { cursor:pointer; font-weight:600; padding:8px 0; }
.videos { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
.video-card { padding:14px; } .video-pair { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
video { width:100%; background:#1e293b; display:block; } button { margin:8px 0; padding:7px 12px; cursor:pointer; }
pre { white-space:pre-wrap; font-size:.85rem; padding:18px; background:#e9edf2; border-radius:8px; max-height:550px; overflow:auto; }
@media(max-width:800px) { main { padding:22px 16px; } .cards { grid-template-columns:1fr 1fr; } .videos { grid-template-columns:1fr; } }
@media print { body { background:white; } nav,button { display:none; } figure,.video-card { break-inside:avoid; } }
</style></head><body><main>"""
    ]
    sections += [
        '<div class="eyebrow">SO-101 · hidden-prong insertion · exploratory validation</div>',
        "<h1>Does moving the camera help plug insertion?</h1>",
        f"<p><strong>The {criterion} favored initial camera positioning with memory in this seed.</strong> "
        f"It succeeded in {100 * headline['initial']['success_rate']:.2f}% of validation episodes, compared with "
        f"{100 * headline['active']['success_rate']:.2f}% for continual active camera motion, "
        f"{100 * headline['wrist_static']['success_rate']:.2f}% for fixed view 7 and "
        f"{100 * headline['wrist']['success_rate']:.2f}% for wrist-only vision. Continual camera motion has no demonstrated "
        "advantage over initial positioning in this experiment. Initial-only freezes camera targets "
        "after 1 s; both camera streams continue updating throughout manipulation.</p>",
        '<div class="notice"><strong>Measurement audit: original scores sample lagged derived state.</strong> '
        "Exact termination-time audits found flagged successes whose current physical positions exceeded 2 mm. "
        "Repeat outcomes also changed despite matching initial state/image hashes. Original scores remain "
        "historical evidence. Corrected headline rates evaluate the existing old-trained checkpoints under the "
        "physical criterion; they are not results of corrected retraining. Both sets are shown separately below.</div>",
        '<div class="notice"><strong>Training completed, but the task is not reliably solved.</strong> '
        "All four policies fail every tested <code>xm</code> episode. Reward improvement and a flat tail "
        "can coexist with a poor or variant-specific policy. Controlled optimizer tuning must be checked "
        "before replication and claims about active perception.</div>",
        '<nav><a href="#training">Training quality</a><a href="#performance">Performance</a>'
        '<a href="#camera">Camera behavior</a><a href="#videos">Videos</a>'
        '<a href="#tuning">Training fixes</a><a href="#limits">Methods & limitations</a></nav>',
        '<div class="cards">',
    ]
    for r in headline_metrics:
        sections.append(
            f'<div class="card"><span>{esc(LABELS[r["condition"]])}</span>'
            f"<strong>{100 * r['success_rate']:.2f}%</strong>"
            f'<span class="small">{r["successes"]}/512 · 95% Wilson '
            f"[{100 * r['wilson95_low']:.2f}, {100 * r['wilson95_high']:.2f}]</span></div>"
        )
    sections += [
        '</div><p class="small">Generated '
        + esc(analysis["generated_at"])
        + f". Cards show the {criterion}. One training seed; intervals describe episode uncertainty. This report loads entirely from "
        "local files and uses the captured evaluated trajectories for its videos.</p>",
        '<div class="downloads">'
        + "".join(
            link(output / name, label)
            for name, label in (
                (
                    "corrected_performance_metrics.csv"
                    if corrected_complete
                    else "performance_metrics.csv",
                    "Performance CSV",
                ),
                (
                    "corrected_performance_metrics.tex"
                    if corrected_complete
                    else "performance_metrics.tex",
                    "LaTeX table",
                ),
                ("evaluation_episodes.csv", "Episode data"),
                ("analysis.json", "Analysis JSON"),
                ("representative_trajectories.csv", "Trajectory diagnostics"),
            )
        )
        + "</div>",
        '<h2 id="training">Training completed; reward alone does not pass the quality gate</h2>',
        "<p>Each condition completed 1,500 contiguous logged updates and 18,432,000 environment transitions "
        "at N512 × 24 rollout steps. The runs resumed their original 100-update pilots, "
        "restoring model and optimizer but resetting simulator, RNG and recurrent episode state. "
        "The dashed line marks this resume. Training success is a native mean over reset batches; "
        "it is not the balanced, episode-weighted evaluation success below.</p>",
        plot(
            "learning_curves",
            "Training learning curves; faint lines are raw updates and solid lines are 30-update moving means.",
        ),
        table(
            [
                "Condition",
                "Reward first 100",
                "Reward last 100",
                "Gain",
                "Tail slope / M",
                "Tail fitted change",
                "Flat by heuristic?",
            ],
            [
                [
                    esc(LABELS[c]),
                    f"{r['first_100_reward_mean']:.3f}",
                    f"{r['last_100_reward_mean']:.3f}",
                    f"{r['reward_gain']:+.3f}",
                    f"{r['last_300_reward_slope_per_million_transitions']:+.3f}",
                    f"{r['last_300_reward_fitted_change']:+.3f}",
                    "Yes" if r["descriptive_reward_plateau"] else "No",
                ]
                for c, r in analysis["convergence"].items()
            ],
        ),
        '<p class="small">The early window is updates 1–100 (the first three have no episode reward). '
        "The tail is updates 1201–1500. “Flat” means both its fitted reward change and the difference "
        "between its two 150-update means are within 10% of max(0.1, |tail mean reward|). "
        "This is a transparent descriptive threshold, not a stationarity test. Autocorrelated training "
        "metrics and a single training seed do not support a confidence claim about convergence.</p>",
        plot(
            "optimization",
            "Adaptive learning rate, exploration scale, critic loss and policy entropy. Mean standard deviation can conceal collapse on individual action axes.",
        ),
        "<p>Learning rate spending long periods at its adaptive floor and narrow action distributions motivate "
        "the controlled LR/entropy continuation. They do not establish that optimization caused the failures. "
        "Reward contains progress shaping, a success bonus and motion penalties; the current telemetry "
        "logs their combined task term, so an independent decomposition of each component cannot be reconstructed.</p>",
    ]
    audit_path = output / "training_audit.json"
    if audit_path.exists():
        sections += [
            "<details><summary>Detailed completion, plateau and success-timing audit</summary>"
            + link(audit_path, "Audit JSON")
            + "<pre>"
            + esc(json.dumps(read(audit_path), indent=2))
            + "</pre></details>"
        ]
    sections += [
        '<h2 id="performance">Balanced validation performance and scoring correction</h2>',
        "<p>Four matched training budgets were evaluated with 512 episodes each: 128 per hidden-prong variant, "
        "N128 and reset seeds 10000–10003. Success requires object-to-goal position error below 2 mm for "
        "three consecutive control samples at 25 Hz. Their first/last endpoints span 0.08 s; the criterion "
        "does not prove continuous dwell between samples. The original evaluator samples lagged derived state; the corrected evaluator "
        "samples current qpos inside the same three-step hold, rather than filtering the last frame of an old success. "
        "The task runs for 3.5 s with an initial 1 s inspection phase; "
        "manipulation starts after inspection. This position-based benchmark criterion is not an independent "
        "force/contact seating measurement.</p>",
        "<p>The <strong>initial-only condition limits camera movement, not image acquisition</strong>. "
        "After 1 s, its joint targets stay fixed while fresh external and wrist images continue arriving "
        "at every control step. The actor retains its GRU state. Testing a true initial snapshot requires "
        "a separate stale-image intervention.</p>",
        '<h3 id="corrected">Corrected criterion · current physical qpos with three-step hold</h3>',
        table(
            [
                "Condition",
                "Success / N",
                "Success % [95% CI]",
                "xm %",
                "xp %",
                "ym %",
                "yp %",
                "Criterion",
            ],
            [
                [
                    link(r["evaluation_report"], LABELS[r["condition"]]),
                    f"{r['successes']}/{r['episodes']}",
                    f"{100 * r['success_rate']:.2f} [{100 * r['wilson95_low']:.2f}, {100 * r['wilson95_high']:.2f}]",
                    *[f"{100 * r[v + '_success_rate']:.2f}" for v in plug.VARIANTS],
                    esc(r["success_state_sample"]),
                ]
                for r in analysis["corrected_performance_metrics"]
            ],
        )
        if analysis["corrected_performance_metrics"]
        else '<p class="small">Corrected evaluations are pending.</p>',
        "<p>"
        + link(output / "corrected_performance_metrics.csv", "Corrected performance CSV")
        + "</p>"
        if analysis["corrected_performance_metrics"]
        else "",
        "<h3>Original recorded results · derived-substep sampling</h3>",
        plot(
            "success_rates",
            "Balanced overall success with Wilson intervals, and per-variant rates at the final checkpoints.",
        ),
        table(
            [
                "Condition",
                "Success / N",
                "Success % [95% CI]",
                "xm %",
                "xp %",
                "ym %",
                "yp %",
                "Success time s",
                "Camera rad",
                "GPU h",
            ],
            [
                [
                    esc(LABELS[r["condition"]]),
                    f"{r['successes']}/{r['episodes']}",
                    f"{100 * r['success_rate']:.2f} [{100 * r['wilson95_low']:.2f}, {100 * r['wilson95_high']:.2f}]",
                    *[f"{100 * r[v + '_success_rate']:.2f}" for v in plug.VARIANTS],
                    f"{r['mean_success_completion_seconds']:.3f}",
                    f"{r['mean_camera_joint_travel_rad']:.3f}",
                    f"{r['training_gpu_hours']:.3f}",
                ]
                for r in metrics
            ],
        ),
        '<p class="small">Success time is conditional on success, so it compares different survivor subsets. '
        "Camera rad is the sum of absolute physical joint changes, with terminal reset jumps excluded. "
        "The tiny wrist/fixed values reflect physical settling rather than commanded camera motion. "
        "GPU hours are measured one-GPU Python wall time across both training segments including W&B shutdown; "
        "they exclude queue and evaluation time. Original reports do not distinguish timeout from failure termination: "
        "the CSV leaves both fields missing instead of equating every unsuccessful episode with timeout.</p>",
        plot(
            "checkpoint_success",
            "Available early/intermediate/final validation scores; missing intermediate initial/active evaluations are not interpolated measurements.",
        ),
        "<p>In the historical derived-state results, initial-only exceeds active by 8.59 percentage points while using about 4.06× less "
        "camera joint travel. Additional camera access is promising relative to these baselines, but view 7 "
        "has not completed a validation-success-based fixed-view search. Independently trained policies "
        "also differ in optimization outcomes, so these numbers are not a causal test of camera movement.</p>",
    ]
    terminal_path = output / "physical_terminal_errors.json"
    native_path = output / "randomized_feasibility_native.json"
    if terminal_path.exists() or native_path.exists():
        sections.append('<h2 id="failure">Failure analysis and sampled native feasibility</h2>')
    if terminal_path.exists():
        terminal = read(terminal_path)
        xm = {
            r["case"]: r
            for r in terminal["rows"]
            if r["stage"] == "original_checkpoints" and r["variant"] == "xm"
        }
        sections += [
            "<h3>XM physical terminal error · corrected original checkpoints</h3>",
            "<p>These population diagnostics use the first terminal raw-qpos object-origin error "
            "from all 128 XM episodes per corrected-condition evaluation. A single terminal "
            "sample below 5 mm is a near-goal diagnostic; success still requires three consecutive "
            "samples below 2 mm. Approaching the goal without meeting that hold remains failure.</p>",
            table(
                [
                    "Condition",
                    "XM held-success / N",
                    "Median terminal error mm",
                    "Single terminal error <5 mm %",
                ],
                [
                    [
                        link(xm[c]["source_report"], LABELS[c]),
                        f"{xm[c]['successes']}/{xm[c]['episodes']}",
                        f"{xm[c]['median_terminal_error_mm']:.2f}",
                        f"{100 * xm[c]['terminal_below_5mm_fraction']:.1f}",
                    ]
                    for c in CONDITIONS
                    if c in xm
                ],
            ),
            '<p class="small">Error norms alone do not identify perception, IK, actuator tracking '
            "or contact as the cause. They measure neither actual prong seating depth nor trajectory "
            "minimum error, and do not establish continuous dwell. Other variants and separate "
            "optimizer/reward executions remain in the full exports.</p>",
            "<p>"
            + link(output / "physical_terminal_errors.csv", "Physical terminal-error CSV")
            + " · "
            + link(terminal_path, "Terminal-error definitions and provenance JSON")
            + "</p>",
        ]
    if native_path.exists():
        native = read(native_path)
        sections += [
            "<h3>Sampled native feasibility · unchanged privileged controller</h3>",
            f"<p>The existing scripted controller succeeds in {native['successes']}/{native['episodes']} "
            f"randomized native episodes: {len(native['seeds'])} seeds paired across four variants. "
            "Scoring uses three current raw-qpos samples below 2 mm, with the same nominal 3.5 s "
            "horizon (88 control steps, quantized timeout 3.52 s). The controller and action bounds "
            "were not changed for this audit.</p>",
            table(
                [
                    "Variant",
                    "Native success / N",
                    "Success %",
                    "Held-success time range s",
                    "Failed seeds",
                ],
                [
                    [
                        esc(v),
                        f"{r['successes']}/{r['episodes']}",
                        f"{100 * r['success_rate']:.1f}",
                        f"{r['held_success_time_min_s']:.2f}–{r['held_success_time_max_s']:.2f}"
                        if r["successes"]
                        else "No success",
                        ", ".join(map(str, r["failed_seeds"])) or "None",
                    ]
                    for v, r in native["per_variant"].items()
                ],
            ),
            '<p class="small">Native float64 IK/physics and privileged object, goal and variant '
            "access are different from the learned Warp visual policy. Identical numeric seed "
            "labels do not pair native initial states with GPU initial states. This supports "
            "feasibility for these sampled native worlds, not all-world solvability, visual "
            "learnability, active-perception utility or a purely optimization-based explanation "
            "of learned failures.</p>",
            "<p>"
            + link(native_path, "Native feasibility source, sampled states and full traces JSON")
            + "</p>",
        ]
        for failed in (r for r in native["rollouts"] if not r["success"]):
            sections.append(
                f'<p class="small">Native failure {esc(failed["variant"])} seed {failed["seed"]}: '
                f"controller remained at approach stage {failed['terminal_controller_stage']}; "
                f"terminal TCP tracking residual {failed['terminal_tcp_tracking_error_mm']:.2f} mm "
                f"and object-origin goal error {failed['terminal_error_mm']:.2f} mm. "
                "All action components stayed within bounds. This sampled failure prevents treating "
                "scripted native success as universal feasibility.</p>"
            )
    gpu_path = output / "randomized_feasibility_gpu.json"
    gpu_verified_path = output / "randomized_feasibility_gpu_verified.json"
    if gpu_path.exists() and gpu_verified_path.exists():
        gpu_verified = read(gpu_verified_path)
        if gpu_verified["status"] == "VERIFIED":
            with gpu_path.open("rb") as source_file:
                assert (
                    hashlib.file_digest(source_file, "sha256").hexdigest()
                    == gpu_verified["artifact_sha256"]
                )
            pairing = gpu_verified["initial_pairing"]
            sections += [
                "<h3>GPU control feasibility · privileged controller, paired initial worlds</h3>",
                f"<p>The unchanged privileged controller achieves {gpu_verified['successes']}/{gpu_verified['episodes']} "
                f"physical held-three successes ({100 * gpu_verified['success_rate']:.2f}%) with actual "
                "Warp control IK and physics. All successes occur at 2.12 s, before the nominal 3.5 s "
                "horizon; nine XM worlds time out at the quantized 3.52 s limit. Current-qpos "
                f"hold violations are {gpu_verified['physical_hold_violations']}; "
                f"{gpu_verified['passed_checks']}/{gpu_verified['total_checks']} independent provenance, "
                "state and scoring checks passed.</p>",
                table(
                    ["Variant", "Privileged GPU success / N", "Success %"],
                    [
                        [
                            esc(v),
                            f"{r['successes']}/{r['episodes']}",
                            f"{100 * r['success_rate']:.2f}",
                        ]
                        for v, r in gpu_verified["per_variant"].items()
                    ],
                ),
                f'<p class="small">The initial conditions match the corrected fixed-view evaluation: '
                f"{pairing['combined_mismatches']} combined-state/image hash mismatches and "
                f"{pairing['physics_mismatches']} physics mismatches, with both sensor streams matched. "
                "This establishes initial pairing, not identical trajectories. Native proxies "
                "forward copied Warp states for privileged float64 FK; they do not step native "
                "physics. Object, goal and variant access are unavailable to the visual actor, so "
                "this is sampled control-feasibility evidence rather than a learned visual baseline, "
                "all-world solvability proof or active-perception benefit. The nine approach-stage "
                "timeouts remain unresolved; controller failure does not prove physical impossibility.</p>",
                "<p>"
                + link(gpu_path, "GPU initial states and terminal physical records")
                + " · "
                + link(gpu_verified_path, "Independent GPU feasibility verification")
                + " · "
                + link(pairing["reference_report"], "Paired fixed-view evaluation")
                + "</p>",
            ]
    sections += [
        '<h2 id="camera">What did the camera do?</h2>',
        "<p>The plots below use the first observed success and first observed failure for each variant where "
        "available. This selection is reproducible and displays failures as well as successes, but it is outcome-selected "
        "and cannot estimate population-average visibility, contacts or final alignment. Initial-only camera targets "
        "should stop changing after 1 s; physical joints can still settle around their frozen target.</p>",
        plot(
            "capture_diagnostics",
            "Before/after inspection joint movement, last captured errors and sampled native prong visibility; circles are successes and crosses are failures.",
        ),
        plot(
            "trace_timecourses",
            "Actual evaluated pre-step state/error captures. The 1 s phase boundary is dotted; the 2 mm criterion is shown for orientation.",
        ),
        plot(
            "camera_trajectories",
            "External-camera Cartesian trajectories from native forward kinematics applied to captured states. Solid: success, dashed: failure; black dot: initial pose.",
        ),
        '<p class="small">Geometric visibility samples 18 points inside the two prongs, checks the camera frustum '
        "and native MuJoCo ray intersections. It is a diagnostic line-of-sight proxy; it does not measure whether "
        "a prong is large enough to resolve in the captured Warp pixels. Contacts are native forward-state "
        "reconstructions, not recorded Warp forces/contact events. These diagnostic fields stay in the selected-trajectory "
        "CSV and are not mixed into the population performance table.</p>",
        table(
            [
                "Condition",
                "Captured N",
                "Post-1s target max change rad",
                "Failure last-error range mm",
                "Camera contact trace steps",
            ],
            [
                [
                    esc(LABELS[c]),
                    str(len(selected)),
                    f"{max(r['camera_target_max_change_post_1s_rad'] or 0 for r in selected):.6f}",
                    f"{min(r['last_preterminal_error_mm'] for r in selected if not r['success']):.1f}–"
                    f"{max(r['last_preterminal_error_mm'] for r in selected if not r['success']):.1f}",
                    str(sum(r["native_reconstructed_camera_contact_steps"] for r in selected)),
                ]
                for c in CONDITIONS
                if (selected := [r for r in captures if r["condition"] == c])
            ],
        ),
        "<p>To determine whether continuing to look helps rather than merely selecting an informative view, "
        "the next comparisons should freeze camera targets, hold images stale, replay camera commands from other "
        "episodes and compare a separately trained scheduled-camera policy. Those interventions and ≥3 training seeds "
        "are necessary before claiming a benefit from feedback-dependent camera motion.</p>",
        '<h2 id="videos">Representative evaluated trajectories</h2>',
        "<p>Each pair shows a full-scene overview and the exact actor camera inputs, captured during the scored "
        "evaluation. The overview is rendered from stored physical states without stepping the simulator. "
        "Use “Play pair” to start both at the same frame; no stochastic physics replay is used. "
        "There is no <code>xm</code> success video because none occurred.</p>",
    ]
    for condition in CONDITIONS:
        sections.append(
            f'<details open><summary>{esc(LABELS[condition])}</summary><div class="videos">'
        )
        for r in (r for r in captures if r["condition"] == condition):
            sections.append(
                f'<article class="video-card"><h3>{esc(r["variant"])} · '
                f"{'success' if r['success'] else 'failure'} · episode {r['episode']}</h3>"
                f'<p class="small">Minimum error {r["minimum_preterminal_error_mm"]:.1f} mm; '
                f"last captured error {r['last_preterminal_error_mm']:.1f} mm; "
                f"camera travel {r['camera_joint_travel_pre_1s_rad'] + r['camera_joint_travel_post_1s_rad']:.2f} rad.</p>"
                '<div class="video-pair">'
            )
            for key, label in (
                ("outside_video", "Scene overview"),
                ("policy_video", "Exact actor inputs"),
            ):
                src = esc(os.path.relpath(root / r[key], output))
                sections.append(
                    f'<div><div class="small">{label}</div>'
                    f'<video controls preload="none" playsinline src="{src}" '
                    f'poster="assets/{r["condition"]}_{r["episode"]}_{key}.jpg"></video></div>'
                )
            sections.append(
                '</div><button type="button" onclick="playPair(this)">Play pair</button> '
                + link(r["trace"], "Captured trace")
                + "</article>"
            )
        sections.append("</div></details>")
    visibility_path = output / "randomized_fixed_view_visibility.json"
    if visibility_path.exists():
        visibility = read(visibility_path)
        records = visibility["records"]
        minimum = min(
            p["external"]["pixels_delta_ge_10"]
            for r in records
            for p in r["pairwise_image_differences"].values()
        )
        absent_ym = sum(
            r["views"]["ym"]["external"]["prong_a_pixels"]
            + r["views"]["ym"]["external"]["prong_b_pixels"]
            == 0
            for r in records
        )
        sections += [
            "<h3>Randomized fixed-view information diagnostic · native renderer</h3>",
            f"<p>Across {len(records)} paired pose-times from 16 reset seeds at 0 and 0.96 s, "
            f"fixed view 7 distinguishes all six variant image pairs in every sample, with at least "
            f"{minimum} pixels differing by ≥10 in a color channel. Matched qpos differences are "
            f"{visibility['summary']['max_matched_qpos_difference']:.1f}. Wrist images do not distinguish "
            f"all four variants at these poses. The <code>ym</code> variant has zero visible external "
            f"prong pixels in {absent_ym}/{len(records)} samples, but appearance/absence still distinguishes "
            "it from the other variants.</p>",
            '<p class="small">This is native geometry/information evidence, not learned classification '
            "or GPU validation. It rejects a blanket claim that fixed view 7 provides no variant information "
            "at these matched poses; it does not establish identification by the trained Warp policy.</p>",
            "<p>" + link(visibility_path, "Paired pose/variant image evidence") + "</p>",
        ]
    if corrected_captures:
        sections += [
            "<h3>Corrected-criterion representative videos</h3>",
            "<p>Captured during the separate current-qpos evaluation of the same trained checkpoints. "
            "These outcomes use the corrected three-step hold; they are not retroactive labels for the "
            'original clips.</p><div class="videos">',
        ]
        for r in corrected_captures:
            sections.append(
                f'<article class="video-card"><h3>{esc(LABELS[r["condition"]])} · '
                f"{esc(r['variant'])} · {'success' if r['success'] else 'failure'} · episode {r['episode']}</h3>"
                '<div class="video-pair">'
            )
            for key, label in (
                ("outside_video", "Scene overview"),
                ("policy_video", "Exact actor inputs"),
            ):
                src = esc(os.path.relpath(root / r[key], output))
                sections.append(
                    f'<div><div class="small">{label}</div><video controls preload="none" '
                    f'playsinline src="{src}" poster="assets/corrected-{r["condition"]}_{r["episode"]}_{key}.jpg"></video></div>'
                )
            sections.append(
                '</div><button type="button" onclick="playPair(this)">Play pair</button></article>'
            )
        sections.append("</div>")
    sections += [
        '<h2 id="tuning">Controlled training fixes and next experiments</h2>',
        "<p>The completed short optimizer screen was inconclusive: fixed LR 1/512, increased entropy 7/512 "
        "and both changes 5/512, versus baseline fixed-view 7/512 (capture repeat 4/512). "
        "The prepared continuations retain each configuration and W&B identity to a matched "
        "9,228,288-transition total (751 updates), compared with the existing baseline at that budget "
        "(44/512). Architecture, geometry and reward remain fixed. The continuation spends 651 additional "
        "updates per candidate, not a new 751-update budget.</p>",
    ]
    physical_tuning = analysis["corrected_hparam_results"]
    sections += [
        "<h3>Matched optimizer checkpoints · corrected physical-state scoring</h3>",
        "<p>Baseline, fixed LR, increased entropy and both changes are compared at model 750 "
        "(751 updates / 9,228,288 training transitions). These are corrected-criterion evaluations "
        "of checkpoints trained under the earlier criterion, with no new optimizer updates. Each "
        "execution has 512 balanced validation episodes (128 per variant), starting at the same "
        "reset seeds. Success requires three consecutive <code>current_qpos</code> samples at 25 Hz. "
        "The bootstrap correction in evaluation source does not retrain these policies.</p>",
    ]
    if physical_tuning:
        sections.append(
            table(
                [
                    "Optimizer / repeat",
                    "Success / N",
                    "Success % [95% CI]",
                    "xm % [95% CI]",
                    "xp % [95% CI]",
                    "ym % [95% CI]",
                    "yp % [95% CI]",
                    "Changed episodes",
                    "Initial hash mismatches",
                    "Physical audit violations",
                ],
                [
                    [
                        link(r["report"], f"{r['case']} / {r['repeat']}"),
                        f"{r['successes']}/{r['episodes']}",
                        f"{100 * r['success_rate']:.2f} [{100 * r['wilson95_low']:.2f}, {100 * r['wilson95_high']:.2f}]",
                        *[
                            f"{100 * r[v + '_success_rate']:.2f} [{100 * r[v + '_wilson95_low']:.2f}, {100 * r[v + '_wilson95_high']:.2f}]"
                            for v in plug.VARIANTS
                        ],
                        str(r["changed_episode_count"])
                        if r["changed_episode_count"] is not None
                        else "Reference repeat",
                        str(r["initial_hash_mismatches"])
                        if r["initial_hash_comparable"]
                        else "Reference / not comparable",
                        "/".join(
                            str(r[k]) if r[k] is not None else "NA"
                            for k in (
                                "success_hold_violations",
                                "success_distance_violations",
                                "success_three_sample_violations",
                                "success_qpos_above_distance_threshold",
                            )
                        ),
                    ]
                    for r in physical_tuning
                ],
            )
        )
        sections += [
            "<p>"
            + link(output / "corrected_hparam_metrics.csv", "Corrected optimizer CSV")
            + " · "
            + link(output / "corrected_hparam_metrics.tex", "Paper LaTeX table")
            + "</p>",
            '<p class="small">Audit columns show hold / distance / three-sample / terminal raw-qpos '
            "violations. CSV includes combined, physics and image hash mismatches. Wilson intervals "
            "describe episode uncertainty for one checkpoint, not training-seed variation. Repeat "
            "executions are numerical diagnostics; they are not pooled into 1,024 independent "
            "episodes or treated as new training seeds. Same initial hashes do not imply matching "
            "trajectories. Partial results remain partial until both repeats of all four cases finish.</p>",
        ]
    else:
        sections.append(
            '<p class="small">No completed matched corrected-criterion optimizer evaluation was found at report generation.</p>'
        )
    sections += [
        "<h3>Historical optimizer scores · lagged derived-state criterion</h3>",
        "<p>The following preserved scores use <code>derived_substep</code> geometry. They are "
        "historical manager flags and must be kept separate from current physical-state success.</p>",
    ]
    if tuning:
        sections.append(
            table(
                ["Completed continuation", "Legacy score / N", "Legacy score %", "Provenance"],
                [
                    [
                        esc(r["case"]),
                        f"{r['successes']}/{r['episodes']}",
                        f"{100 * r['success_rate']:.2f}",
                        link(r["report"], "Evaluation JSON"),
                    ]
                    for r in tuning
                ],
            )
        )
    else:
        sections.append(
            '<div class="notice">No completed matched-budget tuning evaluation was found when this '
            "report was generated. The training-fix outcome is pending. Re-run the report builder after "
            "those evaluations finish.</div>"
        )
    sections += [
        "<h3>Fresh corrected-training reward repair · matched 100-update screen</h3>",
        "<p>This separate pair starts fresh under <code>current_qpos</code> success sampling: "
        "100 updates / 1,228,800 transitions each, seed 0, N512, fixed view 7, GRU, clean occlusion "
        "and matched optimization settings. Progress shaping is compared with a historical-inspired "
        "log-distance objective and stronger three-sample success bonus. Both dense shaping and bonus "
        "magnitude change together. This is an objective-profile comparison, not an isolated shaping "
        "ablation or an exact historical reproduction. It does not replace the 18.432M condition table "
        "or the evaluations of old-trained checkpoints above.</p>",
    ]
    repairs = analysis["reward_repair_results"]
    if repairs:
        sections.append(
            table(
                [
                    "Fresh reward profile",
                    "Total updates",
                    "Total transitions",
                    "Repeat",
                    "Success / N",
                    "Success % [95% CI]",
                    "xm %",
                    "xp %",
                    "ym %",
                    "yp %",
                    "Changed episodes",
                    "Initial hash mismatches",
                ],
                [
                    [
                        link(r["report"], r["profile"]),
                        str(r["training_updates"])
                        if r["training_updates"] is not None
                        else "Unverified",
                        f"{r['training_transitions']:,}"
                        if r["training_transitions"] is not None
                        else "Unverified",
                        str(r["repeat"]),
                        f"{r['successes']}/{r['episodes']}",
                        f"{100 * r['success_rate']:.2f} [{100 * r['wilson95_low']:.2f}, {100 * r['wilson95_high']:.2f}]",
                        *[f"{100 * r[v + '_success_rate']:.2f}" for v in plug.VARIANTS],
                        str(r["changed_episode_count"])
                        if r["changed_episode_count"] is not None
                        else "Reference repeat",
                        str(r["initial_hash_mismatches"])
                        if r["initial_hash_mismatches"] is not None
                        else "Reference repeat",
                    ]
                    for r in repairs
                ],
            )
        )
        sections += [
            "<p>" + link(output / "reward_repair_metrics.csv", "Reward-repair CSV") + "</p>",
            '<p class="small">Each repeat has 512 balanced validation episodes. Repeating evaluation '
            "does not create another training seed; Wilson intervals remain conditional on this single "
            "trained checkpoint. Reward scale changes prevent comparing raw training rewards between "
            "profiles as a success metric.</p>",
        ]
        for case in ("progress", "legacy_log_hold"):
            directory = output / "reward_repair" / case
            training_path = directory / "training.json"
            if training_path.exists():
                sections.append("<p>" + link(training_path, case + " training summary") + "</p>")
            sections += reward_video_cards(root, output, directory, case, "Fresh 100-update screen")
    else:
        sections.append(
            '<p class="small">No completed corrected-training reward-repair evaluation was found '
            "at report generation. The bounded screen is pending; improvement has not been established.</p>"
        )
    rescue_plan = output / "reward_rescue_plan.json"
    if rescue_plan.exists():
        sections.append(
            "<p>" + link(rescue_plan, "Reward-repair protocol and bounded budget") + "</p>"
        )
    sections += [
        "<h3>Matched reward-profile continuation · separate stage and source history</h3>",
        "<p>The prospective target is 751 total updates / 9,228,288 transitions per profile, "
        "retaining the first 100 updates and adding 651. The first stage remains at source "
        "<code>434c37a</code>; a continuation uses a separately frozen source with the same "
        "critic-bootstrap memory correction in both profiles. This is a resumed experiment with "
        "mixed source history. Simulator state, RNG and recurrent episode state are restarted on "
        "resume. More training and the source change occur together, so any improvement cannot be "
        "attributed to the critic correction alone. The table reports actual manifest-derived total "
        "budgets and keeps this stage separate from the fresh screen and the original condition "
        "comparison.</p>",
    ]
    continuations = analysis["reward_repair_continuation_results"]
    if continuations:
        sections.append(
            table(
                [
                    "Resumed reward profile",
                    "Total updates",
                    "Total transitions",
                    "Repeat",
                    "Success / N",
                    "Success % [95% CI]",
                    "xm %",
                    "xp %",
                    "ym %",
                    "yp %",
                    "Added-stage source",
                    "Changed episodes",
                ],
                [
                    [
                        link(r["report"], r["profile"]),
                        str(r["training_updates"])
                        if r["training_updates"] is not None
                        else "Unverified",
                        f"{r['training_transitions']:,}"
                        if r["training_transitions"] is not None
                        else "Unverified",
                        str(r["repeat"]),
                        f"{r['successes']}/{r['episodes']}",
                        f"{100 * r['success_rate']:.2f} [{100 * r['wilson95_low']:.2f}, {100 * r['wilson95_high']:.2f}]",
                        *[f"{100 * r[v + '_success_rate']:.2f}" for v in plug.VARIANTS],
                        esc((r["training_source_revision"] or "Unverified")[:12]),
                        str(r["changed_episode_count"])
                        if r["changed_episode_count"] is not None
                        else "Reference repeat",
                    ]
                    for r in continuations
                ],
            )
        )
        sections.append(
            "<p>"
            + link(
                output / "reward_repair_continuation_metrics.csv",
                "Separate reward-continuation CSV",
            )
            + "</p>"
        )
        for case in ("progress", "legacy_log_hold"):
            directory = output / "reward_repair/continuation" / case
            for filename, label in (
                ("run_manifest.json", "source history and resume manifest"),
                ("training.json", "training summary"),
            ):
                path = directory / filename
                if path.exists():
                    sections.append("<p>" + link(path, case + " " + label) + "</p>")
            completed_case = next((row for row in continuations if row["profile"] == case), None)
            if completed_case:
                updates = completed_case["training_updates"]
                stage_label = (
                    f"Resumed continuation · {updates} total updates"
                    if updates is not None
                    else "Resumed continuation · budget unverified"
                )
                sections += reward_video_cards(root, output, directory, case, stage_label)
    else:
        sections.append(
            '<p class="small">No completed continuation evaluation was found at report generation. '
            "A prepared plan does not establish submission, completion or improvement.</p>"
        )
    completion_path = output / "reward_repair/continuation/completion_verified.json"
    if completion_path.exists():
        sections.append(
            "<p>"
            + link(completion_path, "Independent completion, physical-score and optimization audit")
            + "</p>"
        )
    repair_learning = analysis["reward_repair_learning"]
    if repair_learning:

        def repair_number(row, key):
            return f"{row[key]:.3f}" if row[key] is not None else "Pending"

        sections += [
            "<h3>Reward-repair learning trajectories and descriptive plateau checks</h3>",
            "<p>Each reward objective has its own figure and numerical scale. A reward increase "
            "across profiles is not a comparable performance measure. Completed continuations "
            "display all recorded updates from 1 through 751, retaining the original first stage; "
            "while training is incomplete, only the immutable screen budget is plotted. Budgets "
            "come from stage manifests and cumulative telemetry, not the resumed runner config.</p>",
            table(
                [
                    "Reward profile",
                    "Recorded / target updates",
                    "Reward window",
                    "First mean",
                    "Last mean",
                    "Reward gain",
                    "Tail-300 slope / M",
                    "Fitted tail change",
                    "Last150 - prior150",
                    "Plateau tolerance",
                    "Descriptive plateau",
                    "Tail episode-weighted training success",
                ],
                [
                    [
                        esc(r["profile"]),
                        f"{r['plotted_updates']}/{r['expected_total_updates']} · {esc(r['plotted_stage'])}",
                        f"{r['reward_comparison_window_updates']} updates",
                        repair_number(r, "first_window_reward_mean"),
                        repair_number(r, "last_window_reward_mean"),
                        repair_number(r, "reward_gain"),
                        repair_number(r, "last_300_reward_slope_per_million_transitions"),
                        repair_number(r, "last_300_reward_fitted_change"),
                        repair_number(r, "final_150_vs_previous_150_reward_change"),
                        repair_number(r, "descriptive_plateau_tolerance"),
                        str(r["descriptive_reward_plateau"])
                        if r["tail_300_available"]
                        else "Fewer than 300 completed updates",
                        f"{100 * r['tail_episode_weighted_training_success']:.2f}%"
                        if r["tail_episode_weighted_training_success"] is not None
                        else "Not recorded",
                    ]
                    for r in repair_learning
                ],
            ),
            '<p class="small">The descriptive plateau requires both the fitted tail-300 reward '
            "change and the last150-minus-prior150 mean change to stay within 10% of "
            "max(0.1, absolute tail mean). This heuristic is not a statistical convergence test or "
            "a competence gate. Episode-weighted success sums completed successes and episodes "
            "over the displayed tail; it remains a training statistic. Reward growth and a plateau "
            "must be interpreted alongside corrected heldout per-variant scores.</p>",
            "<p>"
            + link(output / "repair_learning_curves.csv", "Raw reward-repair learning CSV")
            + " · "
            + link(output / "repair_learning_summary.json", "Budget and plateau diagnostics JSON")
            + "</p>",
        ]
        for row in repair_learning:
            sections.append(
                plot(
                    "repair_learning_" + row["profile"],
                    f"{row['profile']}: {row['plotted_updates']} actual recorded updates; reward scale is profile-specific. Raw traces and 30-update means; success smoothing is weighted by completed episodes. Per-action std, KL, learning rate and clipping expose optimization behavior.",
                )
            )
    sections += [
        "<h3>Optional fresh joint exploration repair · separate experiment</h3>",
        "<p>This conditional next step uses one explicitly selected reward profile and starts "
        "fresh with the critic-bootstrap correction from its first update, with no mid-run source "
        "change. Its target is 751 updates / 9,228,288 transitions, plug, fixed view 7, GRU, "
        "<code>current_qpos</code>, N512 and training seed 0. Initial action std 1 and entropy "
        "coefficient 0.01 change together, with adaptive LR 0.0003. This tests a joint exploration "
        "configuration, not an isolated standard-deviation ablation or an exact historical "
        "reproduction. It stays separate from the original reward screen and both resumed reward "
        "profiles. Preparation alone does not establish a submitted run or improved competence.</p>",
    ]
    exploration = analysis["exploration_repair_results"]
    exploration_learning = analysis["exploration_repair_learning"]
    exploration_status = analysis["exploration_repair_status"]
    if exploration_status:
        sections += [
            table(
                [
                    "Selected fresh profile",
                    "Recorded manifest",
                    "Training runtime",
                    "Training job",
                    "Evaluation job",
                    "Target updates",
                    "Target transitions",
                    "Source",
                ],
                [
                    [
                        link(r["manifest"], r["profile"]),
                        esc(r["recorded_manifest_status"] or "Unknown"),
                        esc(r["recorded_training_runtime_status"] or "Not recorded"),
                        esc(str(r["training_job_id"] or "Not recorded")),
                        esc(str(r["evaluation_job_id"] or "Not recorded")),
                        str(r["target_updates"]),
                        f"{r['target_transitions']:,}"
                        if r["target_transitions"] is not None
                        else "Unverified",
                        esc((r["source_revision"] or "Unverified")[:12]),
                    ]
                    for r in exploration_status
                ],
            ),
            '<p class="small">This is the recorded reservation/runtime status at report generation, '
            "not a new live cluster query. Completed training curves and heldout scores appear only "
            "when their actual output artifacts are available.</p>",
        ]
    if exploration:
        sections.append(
            table(
                [
                    "Fresh profile / repeat",
                    "Total updates",
                    "Total transitions",
                    "Success / N",
                    "Success % [95% CI]",
                    "xm %",
                    "xp %",
                    "ym %",
                    "yp %",
                    "Changed episodes",
                    "Initial hash mismatches",
                    "Physical audit violations",
                ],
                [
                    [
                        link(r["report"], f"{r['profile']} / {r['repeat']}"),
                        str(r["training_updates"]),
                        f"{r['training_transitions']:,}"
                        if r["training_transitions"] is not None
                        else "Unverified",
                        f"{r['successes']}/{r['episodes']}",
                        f"{100 * r['success_rate']:.2f} [{100 * r['wilson95_low']:.2f}, {100 * r['wilson95_high']:.2f}]",
                        *[f"{100 * r[v + '_success_rate']:.2f}" for v in plug.VARIANTS],
                        str(r["changed_episode_count"])
                        if r["changed_episode_count"] is not None
                        else "Reference repeat",
                        str(r["initial_hash_mismatches"])
                        if r["initial_hash_comparable"]
                        else "Reference / not comparable",
                        "/".join(
                            str(r[k]) if r[k] is not None else "NA"
                            for k in (
                                "success_hold_violations",
                                "success_distance_violations",
                                "success_three_sample_violations",
                                "success_qpos_above_distance_threshold",
                            )
                        ),
                    ]
                    for r in exploration
                ],
            )
        )
        sections += [
            "<p>"
            + link(output / "exploration_repair_metrics.csv", "Separate exploration CSV")
            + " · "
            + link(output / "exploration_repair_metrics.tex", "Exploration paper LaTeX")
            + "</p>",
            '<p class="small">Each repeat has 512 balanced validation episodes and physical-state '
            "audit counts in hold / distance / three-sample / raw-qpos order. CSV includes per-variant "
            "Wilson intervals, separate physics/image hashes and configuration provenance. Repeats "
            "remain separate numerical executions of one trained seed; they are not pooled. "
            "These fixed-camera results cannot establish the value of active camera movement.</p>",
        ]
        for case in sorted({r["profile"] for r in exploration}):
            directory = output / "exploration_repair" / case
            for filename in ("run_manifest.json", "training.json"):
                path = directory / filename
                if path.exists():
                    sections.append("<p>" + link(path, case + " fresh " + filename) + "</p>")
            sections += reward_video_cards(
                root, output, directory, case, "Fresh joint exploration repair · 751 updates"
            )
    else:
        sections.append(
            '<p class="small">Completed exploration-repair validation is pending; no performance result is inferred from the recorded reservation.</p>'
            if exploration_status
            else '<p class="small">Conditional preparation: no completed exploration-repair validation was found at report generation. No submission or performance result is inferred from the plan.</p>'
        )
    if exploration_learning:
        sections += [
            "<p>"
            + link(
                output / "exploration_learning_curves.csv",
                "All recorded fresh exploration learning rows",
            )
            + " · "
            + link(
                output / "exploration_learning_summary.json",
                "Fresh exploration plateau diagnostics",
            )
            + "</p>",
            '<p class="small">These curves appear only after the full manifest budget is completed '
            "and history is contiguous. Reward scales remain profile-specific. Tail-300 plateau "
            "diagnostics use the descriptive heuristic above and are not evidence of competence.</p>",
        ]
        for row in exploration_learning:
            plateau = (
                str(row["descriptive_reward_plateau"])
                if row["tail_300_available"]
                else "Unavailable"
            )
            sections.append(
                plot(
                    "exploration_learning_" + row["profile"],
                    f"Fresh {row['profile']}: {row['plotted_updates']} actual updates; reward gain {row['reward_gain']:.3f} (first/last {row['reward_comparison_window_updates']} updates); descriptive tail-300 plateau {plateau}. Raw reward, completed-episode-weighted success, KL, per-action std, LR and clipping; no resumed stage is mixed into these curves.",
                )
            )
    credit_path = output / "recurrent_credit_audit.json"
    if credit_path.exists():
        credit = read(credit_path)
        verified = credit["distinct_critic_bootstrap_correction"]["verification"]["passed"]
        sections += [
            "<h3>Recurrent-state diagnosis · confirmed bookkeeping issue and untested credit hypothesis</h3>",
            "<p>A pinned upstream bootstrap call advances the persistent recurrent critic state; "
            "the next collector action consumes the same observation again. The actor is unaffected "
            "by that bootstrap call. The separate correction snapshots and restores critic memory "
            "after computing returns, including on exceptions. Bootstrap values, returns and "
            f"advantages are unchanged for that call; {verified} focused tests passed. This correction "
            "is absent from the original runs and the fresh reward screen. Its effect on learning "
            "has not been measured independently.</p>",
            "<p>The current recurrent gradient window is 24 control steps (0.96 s), shorter than "
            "the 1 s inspection period. That is a plausible credit-assignment limitation, not an "
            "established cause of poor training. Inference memory persists across collector windows, "
            "GAE bootstraps through critic values, and external images continue updating after "
            "inspection. A 24-versus-48-step experiment is prepared conceptually, but remains "
            "<code>PREPARED_NOT_EXECUTABLE</code> pending rollout configuration and transition/resume "
            "accounting changes. No longer-rollout result is claimed.</p>",
            "<p>" + link(credit_path, "Recurrent-credit and critic-bootstrap evidence") + "</p>",
        ]
    for path, label in (
        ("artifacts/hparam_search/next_stage_plan.json", "Prepared tuning protocol"),
        ("artifacts/hparam_search/continuation/manifest.json", "Continuation submissions"),
        ("artifacts/hparam_search/continuation/launch_manifest.json", "Continuation launch record"),
        ("artifacts/dynamic_occlusion/validation.json", "Dynamic-occlusion validation"),
    ):
        if (root / path).exists():
            sections.append("<p>" + link(path, label) + "</p>")
    if legacy:
        sections += [
            "<h3>Recovered historical training: a different configuration</h3>",
            "<p>Saved configs and checkpoints for the earlier <code>pl3</code> runs were recovered. "
            "Their saved training success metrics are high, but their reward, hold criterion, architecture, "
            "control timing and budget differ. They are useful for diagnosing training; they cannot be "
            "inserted as matched held-out results in the current performance table.</p>",
            table(
                [
                    "Saved run",
                    "Training success metric",
                    "Final checkpoint",
                    "Global transitions from saved config",
                ],
                [
                    [
                        esc(r["run_name"]),
                        f"{100 * r['saved_training_summary'].get('Episode_Metrics/success_rate', 0):.2f}%",
                        link(r["final_checkpoint"]["path"], "model_1999.pt"),
                        f"{r['runner']['max_iterations'] * r['runner']['num_steps_per_env'] * r['environment']['num_envs_per_rank'] * r['runner'].get('multi_gpu', {}).get('world_size', 1):,}",
                    ]
                    for r in legacy["records"]
                ],
            ),
            "<p>These saved configurations use a feedforward 256/256/128 network, initial action standard "
            "deviation 1, adaptive LR 0.001, entropy 0.005, five PPO epochs/four minibatches, a 5 s horizon "
            "and 0.022 s control steps. The saved world size is four, so 49.152 million transitions per rank "
            "correspond to 196.608 million globally if that saved distributed setup ran through its target. "
            "The current pilots use 18.432 million. The historical reward gives 1000 at instantaneous "
            "error below 2 mm; the current reward gives 10 after a three-step hold. Those scales and criteria "
            "are not interchangeable. Current variants also share inertias; possible historical nonvisual "
            "variant cues require review.</p>",
            "<p>"
            + link(
                "artifacts/plug_analysis/legacy_run_inventory.json",
                "Recovered historical configs and hashes",
            )
            + "</p>",
        ]
    sections += [
        "<h3>Exact success timing and repeatability</h3>",
        "<p>In pinned MjLab 1.4, termination/reward calculation precedes the final forward update. "
        "Derived object positions at the manager therefore lag raw qpos by one 2 ms physics substep. "
        "The captured images/errors are post-forward, pre-action samples. A success clip ending above "
        "2 mm does not alone disprove the recorded success flag. Instrumented repeats inspect the exact "
        "manager samples and three-step hold without changing physics or random draws. Original scored "
        "results remain in the primary table.</p>",
    ]
    if analysis["termination_audits"]:
        sections.append(
            table(
                [
                    "Diagnostic repeat",
                    "Criterion",
                    "Success / N",
                    "Timeout flags",
                    "Failure flags",
                    "Hold / distance / 3-sample violations",
                    "Current qpos outside 2 mm at legacy flag",
                    "Changed episodes",
                ],
                [
                    [
                        link(r["report"], LABELS[r["condition"]]),
                        esc(r["success_state_sample"]),
                        f"{r['successes']}/{r['episodes']}",
                        str(r["termination_audit"]["terminal_flags"]["timeout"]),
                        str(r["termination_audit"]["terminal_flags"]["failure"]),
                        " / ".join(
                            str(r["termination_audit"].get(k))
                            for k in (
                                "success_hold_violations",
                                "success_distance_violations",
                                "success_three_sample_violations",
                            )
                        ),
                        str(r["termination_audit"].get("success_qpos_above_distance_threshold")),
                        str(
                            (r["repeat_comparison"] or {}).get(
                                "changed_episode_count", "No reference"
                            )
                        ),
                    ]
                    for r in analysis["termination_audits"]
                ],
            )
        )
    else:
        sections.append(
            '<p class="small">Exact termination-time diagnostic repeat results are pending.</p>'
        )
    dynamic_path = root / "artifacts/dynamic_occlusion/summary.json"
    if dynamic_path.exists():
        dynamic = read(dynamic_path)
        sections += [
            "<h3>Dynamic occlusion implementation and scenario validation</h3>",
            "<p>Moving world-space occluders were implemented and checked with task-appropriate timing. "
            "The media below are scripted native feasibility/visibility diagnostics, including privileged "
            "camera schedules. They do not establish learned-policy success or an active-perception benefit.</p>",
            "<p>"
            + link(dynamic_path, "Occlusion validation summary")
            + " · "
            + link("artifacts/dynamic_occlusion/visibility.csv", "Visibility measurements")
            + "</p>",
            table(
                [
                    "Task",
                    "Diagnostic camera",
                    "Useful visibility: panel %",
                    "Same state, panel hidden %",
                ],
                [
                    [
                        esc(r["task"]),
                        esc(r["camera"]),
                        f"{100 * r['useful_visible_fraction_panel']:.1f}",
                        f"{100 * r['useful_visible_fraction_panel_hidden']:.1f}",
                    ]
                    for r in dynamic["rendered_visibility_summary"]
                    if r["snapshot_set"] == "rollout_samples"
                ],
            ),
            "<p>The panel blocks fixed/external views while wrist useful visibility remains 100% in the "
            "pushing samples and 93.5% in transfer with or without the panel. This scenario provides optical "
            "obstruction but does not yet require camera motion. Wrist access, waiting, memory and a searched "
            "fixed view remain essential controls.</p>",
            "<details><summary>Scenario validation measurements and limits</summary><pre>"
            + esc(json.dumps(dynamic, indent=2))
            + '</pre></details><div class="videos">',
        ]
        gpu_path = root / "artifacts/dynamic_occlusion/gpu_validation.json"
        if gpu_path.exists():
            gpu = read(gpu_path)
            sections.append(
                "<p><strong>Six CUDA tests passed:</strong> rendered RGB obstruction, moving-panel "
                "midpoint parity, partial resets and existing pipeline regressions. "
                + link(gpu_path, "GPU validation evidence")
                + "</p>"
                if gpu.get("pytest_passed")
                else "<p>CUDA validation has not passed; see "
                + link(gpu_path, "GPU evidence")
                + ".</p>"
            )
        for r in dynamic.get("media", []):
            if not r.get("video"):
                continue
            base = root / "artifacts/dynamic_occlusion"
            video = esc(os.path.relpath(base / r["video"], output))
            poster = (
                (' poster="' + esc(os.path.relpath(base / r["snapshot"], output)) + '"')
                if r.get("snapshot")
                else ""
            )
            sections.append(
                f'<article class="video-card"><h3>{esc(r.get("task", "Task"))} · scripted dynamic occlusion</h3>'
                f'<video controls preload="none" playsinline src="{video}"{poster}></video></article>'
            )
        sections.append("</div>")
    sections += [
        "<p>After learning and measurement pass the quality gate, compare clean, static and dynamic "
        "occlusion with matched world-space occluder distributions and task-appropriate timings. "
        "Transfer and pushing begin as bounded pilots; pushing can serve as a negative control. "
        "Unpredictable object disturbances should be analyzed separately from image obstruction because "
        "they change whether remembered state remains useful.</p>",
        '<h2 id="limits">Methods, provenance and limits</h2><ul>',
    ]
    sections += ["<li>" + esc(s) + "</li>" for s in analysis["limitations"]]
    sections += [
        "<li>Camera access/action dimensions differ between conditions. Parameter count, camera-motion costs "
        "and learning outcomes can confound a condition comparison.</li>",
        "<li>Early pilot repeats disagreed on a small number of episodes. The final repeat audit is materially "
        "different: 86/512 initial-only and 125/512 active outcomes changed (16.8% and 24.4%), despite identical "
        "initial physical-state/image hashes. Similar total success counts therefore conceal substantial "
        "episode disagreement. Original scored evaluations are preserved; tightly paired intervention claims "
        "need repeated controls and initial-state checks.</li>",
        "</ul>",
        table(
            ["Condition", "Training", "Evaluation", "Checkpoint / report"],
            [
                [
                    esc(LABELS[r["condition"]]),
                    f'<a href="{esc(r["training_wandb_url"])}">W&B training</a>',
                    f'<a href="{esc(r["evaluation_wandb_url"])}">W&B evaluation</a>',
                    link(r["evaluation_report"], "Scored JSON")
                    + " · "
                    + link(r["checkpoint"], "model_1499.pt"),
                ]
                for r in metrics
            ],
        ),
        "<p>"
        + link(
            "artifacts/cluster_pilot/evaluation_repeat_audit.json", "Earlier repeatability audit"
        )
        + " · "
        + link("docs/HANDOFF_2026-10-07.md", "Original handoff")
        + " · "
        + link("PROGRESS.md", "Progress log")
        + "</p>",
        '<p class="small">Rebuild: <code>.venv/bin/python scripts/build_plug_report.py --validate-media</code>. '
        "CSV rates are fractions; displayed HTML/LaTeX rates are percentages. Large checkpoints, captured NPZs "
        "and videos remain local outside Git; the report links to their existing locations.</p>",
        '</main><script>function playPair(button) { const videos=button.closest("article").querySelectorAll("video"); '
        "videos.forEach(v=>{v.pause();v.currentTime=0;});videos.forEach(v=>v.play().catch(()=>{})); }"
        "</script></body></html>",
    ]
    (output / "index.html").write_text("\n".join(sections))


def validate_report(output, root, rows, decode):
    from html.parser import HTMLParser

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.urls = []

        def handle_starttag(self, tag, attrs):
            self.urls.extend(v for k, v in attrs if k in ("href", "src", "poster") and v)

    parser = Links()
    parser.feed((output / "index.html").read_text())
    local = [u for u in parser.urls if not u.startswith(("https://", "#"))]
    missing = [u for u in local if not (output / u).exists()]
    assert not missing, missing
    media = []
    reused = False
    previous_path = output / "report_validation.json"
    if not decode and previous_path.exists():
        previous = read(previous_path).get("videos", [])
        if len(previous) == 2 * len(rows) and all(
            (root / r["path"]).exists()
            and r.get("bytes") == (root / r["path"]).stat().st_size
            and r.get("mtime_ns") == (root / r["path"]).stat().st_mtime_ns
            for r in previous
        ):
            media = previous
            reused = True
    if decode:
        import imageio.v2 as imageio

        for row in rows:
            for key in ("outside_video", "policy_video"):
                path = root / row[key]
                reader = imageio.get_reader(path)
                try:
                    n = reader.count_frames()
                    meta = reader.get_meta_data()
                    for index in (0, n // 2, n - 1):
                        frame = reader.get_data(index)
                        assert frame.ndim == 3 and frame.shape[-1] == 3
                    assert n == row["captured_steps"], (str(path), n, row["captured_steps"])
                    assert abs(meta["fps"] - 25) < 1e-6
                    media.append(
                        {
                            "path": row[key],
                            "frames": n,
                            "fps": meta["fps"],
                            "size": list(meta["size"]),
                            "decoded_first_middle_last": True,
                            "bytes": path.stat().st_size,
                            "mtime_ns": path.stat().st_mtime_ns,
                        }
                    )
                finally:
                    reader.close()
    return {
        "local_links_checked": len(local),
        "missing_links": missing,
        "media_validated": len(media),
        "media_validation_scope": "Original and corrected four-condition representative videos only; linked reward-repair and dynamic videos are covered by the independent portable-bundle validation.",
        "media_decoded_this_build": 0 if reused else len(media),
        "media_validation_reused_unchanged_files": reused,
        "videos": media,
        "html_parsed": True,
    }


def bundle_report(root, output):
    """Package explicitly referenced report files; never crawl the workspace."""
    import hashlib

    allowed = {".html", ".svg", ".pdf", ".jpg", ".png", ".csv", ".tex", ".json", ".md", ".mp4"}
    source = (output / "index.html").read_text()
    sources = set()
    omitted = []
    for target in re.findall(r'(?:href|src|poster)="([^"]+)"', source):
        target = html.unescape(target)
        if target.startswith(("https://", "#")):
            continue
        path = (output / target).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in allowed:
            omitted.append(target)
            continue
        if path.is_file():
            sources.add(path)
            if path.suffix == ".mp4":
                metadata = path.parent / "representatives.json"
                if metadata.exists():
                    sources.add(metadata)
                traces = path.parent.with_name(path.parent.name.removesuffix("-videos") + "-traces")
                capture = traces / "capture.json"
                if capture.exists():
                    sources.add(capture)
    # Compact local measurements are useful independently of the rendered page.
    for pattern in ("*.csv", "*.tex", "*.json", "README.md"):
        sources.update(
            p.resolve()
            for p in output.glob(pattern)
            if p.name not in {"bundle_manifest.json", "portable_validation.json"}
        )
    for target in omitted:
        escaped = html.escape(target)
        source = re.sub(
            r'<a href="' + re.escape(escaped) + r'">(.*?)</a>',
            r'<span class="small">\1 (local research artifact omitted)</span>',
            source,
        )
    source = source.replace(
        "</main>",
        '<p class="small">Portable bundle: plots, tables, scored JSONs and referenced videos are included. '
        "Checkpoints and raw NPZ captures stay in the original research workspace.</p></main>",
    )
    index = output / "index.html"
    sources.discard(index.resolve())
    relative_report = index.relative_to(root).as_posix()
    redirect = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={relative_report}">'
        f'<title>Plug insertion report</title><a href="{relative_report}">Open report</a></html>'
    )
    archive_path = output / "plug-report.zip"
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "archive": str(archive_path.relative_to(root)),
        "entrypoint": "index.html",
        "report": relative_report,
        "omitted_local_links": omitted,
        "files": [],
    }
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("index.html", redirect)
        archive.writestr(relative_report, source)
        for path in sorted(sources):
            assert path.is_relative_to(root) and path.suffix.lower() in allowed
            name = path.relative_to(root).as_posix()
            assert not name.startswith("logs/") and "/.env" not in name
            data = path.read_bytes()
            manifest["files"].append(
                {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
            archive.writestr(
                name,
                data,
                compress_type=zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED,
            )
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
        assert archive.testzip() is None
    manifest["archive_bytes"] = archive_path.stat().st_size
    write_json(output / "bundle_manifest.json", manifest)
    print(f"Portable report bundle: {archive_path} ({archive_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
