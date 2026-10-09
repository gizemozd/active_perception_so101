"""Rendered optical-sweep controls and native scripted feasibility, not learning.

Compare each rendered state to the same state with only the panel hidden. Include
fixed and wrist views as useful alternative strategies; sweep clearance permits
waiting, and unchanged plug geometry can be remembered after inspection.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from .config import TASKS, Experiment, static_candidates
from .native import NativeEnv, ScriptedPolicy
from .occlusion import with_occlusion
from .sanity import CameraAudit, mosaic

CAMERAS = ("manipulator/wrist_cam", "fixed", "camera_arm/wrist_cam")
CAMERA_LABELS = ("wrist", "fixed", "scripted moving external")


def useful_pixels(task, metrics):
    if task == "plug":
        return metrics["prong_a_pixels"] + metrics["prong_b_pixels"]
    return metrics["object_pixels"]


def paired_views(env, audit):
    """Actual segmentation at policy resolution; render a matched counterfactual."""
    panel_position = env.data.mocap_pos[env.panel_id].copy()
    occluded = [audit.render(camera) for camera in CAMERAS]
    try:
        env.data.mocap_pos[env.panel_id, 2] = -1
        mujoco.mj_forward(env.model, env.data)
        clear = [audit.render(camera) for camera in CAMERAS]
    finally:
        env.data.mocap_pos[env.panel_id] = panel_position
        mujoco.mj_forward(env.model, env.data)
    measurements = {}
    for label, (_, current), (_, counterfactual) in zip(
        CAMERA_LABELS, occluded, clear, strict=True
    ):
        measurements[label] = {
            "occluded": current,
            "panel_hidden_counterfactual": counterfactual,
            "object_pixels_removed": counterfactual["object_pixels"] - current["object_pixels"],
            "useful_pixels_removed": useful_pixels(env.cfg.task, counterfactual)
            - useful_pixels(env.cfg.task, current),
            "useful_visible_occluded": useful_pixels(env.cfg.task, current) >= 3,
            "useful_visible_counterfactual": useful_pixels(env.cfg.task, counterfactual) >= 3,
        }
    return measurements, [frame for frame, _ in occluded], [frame for frame, _ in clear]


def clean_counterpart(cfg):
    return with_occlusion(cfg, "clean")


def sensor_comparison(current, clear):
    rows = (
        mosaic(current, [f"panel: {label}" for label in CAMERA_LABELS], scale=2),
        mosaic(clear, [f"same state, panel hidden: {label}" for label in CAMERA_LABELS], scale=2),
    )
    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)))
    for i, row in enumerate(rows):
        canvas.paste(row, (0, i * row.height))
    return canvas


def run_episode(cfg, output, *, stride=3, record_video=False):
    env = NativeEnv(cfg)
    clean = NativeEnv(clean_counterpart(cfg))
    policy = ScriptedPolicy(env)
    audit = CameraAudit(env)
    output.mkdir(parents=True, exist_ok=True)
    basename = f"{cfg.task}_seed{cfg.seed}"
    overview = mujoco.Renderer(env.model, width=640, height=360) if record_video else None
    writer = (
        imageio.get_writer(
            output / f"{basename}.mp4",
            fps=1 / (cfg.step_dt * stride),
            macro_block_size=2,
            codec="libx264",
            quality=7,
        )
        if record_video
        else None
    )
    records = []
    worst_score, worst_image = -1, None
    exact_physics = True
    hold, ever_held = 0, False
    positions = []
    success_steps = []
    # Additional static-state sweep distinguishes absence of a cue at insertion
    # from genuine panel obstruction. This is optical evidence, not a task metric.
    reset_sweep = []
    try:
        from .native import occluder_position

        original = env.data.mocap_pos[env.panel_id].copy()
        for u in np.linspace(0, 1, 9):
            t = env.onset + u * env.duration
            env.data.mocap_pos[env.panel_id] = occluder_position(
                t,
                env.onset,
                env.duration,
                env.side,
                cfg,
                center=env.panel_center,
                travel=env.panel_travel,
            )
            mujoco.mj_forward(env.model, env.data)
            metrics, current, counterfactual = paired_views(env, audit)
            reset_sweep.append({"sweep_fraction": float(u), "views": metrics})
            score = sum(row["object_pixels_removed"] for row in metrics.values())
            if score > worst_score:
                worst_score = score
                worst_image = sensor_comparison(current, counterfactual)
        env.data.mocap_pos[env.panel_id] = original
        mujoco.mj_forward(env.model, env.data)
        for step in range(math.ceil(cfg.episode_seconds / cfg.step_dt)):
            action = policy()
            success = env.step(action)
            clean.step(action)
            exact_physics &= np.array_equal(env.data.qpos, clean.data.qpos)
            hold = hold + 1 if success else 0
            if hold == 3:
                success_steps.append(step + 1)
            ever_held |= hold >= 3
            cid = env.model.camera("camera_arm/wrist_cam").id
            positions.append(env.data.cam_xpos[cid].copy())
            if step % stride:
                continue
            measurements, current, counterfactual = paired_views(env, audit)
            # Panel was positioned before this integration step, exactly as in
            # training; report both control time and rendered-state time.
            records.append(
                {
                    "step": step + 1,
                    "time_s": float(env.data.time),
                    "panel_control_time_s": step * cfg.step_dt,
                    "panel_position_m": env.data.mocap_pos[env.panel_id].tolist(),
                    "panel_present": bool(env.data.mocap_pos[env.panel_id, 2] > 0),
                    "instantaneous_success": bool(success),
                    "success_hold_steps": hold,
                    "views": measurements,
                }
            )
            if writer:
                comparison = sensor_comparison(current, counterfactual)
                overview.update_scene(env.data, camera="overview", scene_option=audit.option)
                image = Image.fromarray(overview.render()).resize(
                    (comparison.width, comparison.width * 360 // 640)
                )
                canvas = Image.new("RGB", (comparison.width, image.height + comparison.height))
                canvas.paste(image, (0, 0))
                canvas.paste(comparison, (0, image.height))
                ImageDraw.Draw(canvas).text(
                    (8, 8),
                    f"{cfg.task} / t={env.data.time:.2f}s / privileged scripted control",
                    fill="white",
                    stroke_width=1,
                    stroke_fill="black",
                )
                writer.append_data(np.asarray(canvas))
        worst_image.save(output / f"{basename}_reset_obstruction.png")
    finally:
        if writer:
            writer.close()
        if overview:
            overview.close()
        audit.close()
    travel = float(np.linalg.norm(np.diff(np.asarray(positions), axis=0), axis=1).sum())
    return {
        "experiment": cfg.to_dict(),
        "sampled_panel": {
            "onset_s": env.onset,
            "duration_s": env.duration,
            "direction": env.side,
            "center_m": env.panel_center.tolist(),
            "travel_m": env.panel_travel,
        },
        "scripted_success_held_3_steps": bool(ever_held),
        "scripted_first_success_time_s": success_steps[0] * cfg.step_dt if success_steps else None,
        "final_distance_m": float(np.linalg.norm(env.object_pos - env.goal)),
        "matched_clean_dynamic_qpos_exactly_equal_every_step": bool(exact_physics),
        "camera_cartesian_travel_m": travel,
        "reset_sweep": reset_sweep,
        "rollout_samples": records,
        "video": f"{basename}.mp4" if record_video else None,
        "snapshot": f"{basename}_reset_obstruction.png",
    }


def summarize(episodes):
    rows = []
    for task in sorted({row["experiment"]["task"] for row in episodes}):
        selected = [e for e in episodes if e["experiment"]["task"] == task]
        for kind in ("reset_sweep", "rollout_samples"):
            samples = [
                r for e in selected for r in e[kind] if kind == "reset_sweep" or r["panel_present"]
            ]
            for camera in CAMERA_LABELS:
                views = [r["views"][camera] for r in samples]
                rows.append(
                    {
                        "task": task,
                        "snapshot_set": kind,
                        "camera": camera,
                        "episodes": len(selected),
                        "rendered_samples": len(views),
                        "mean_object_pixels_removed": float(
                            np.mean([v["object_pixels_removed"] for v in views])
                        )
                        if views
                        else None,
                        "maximum_object_pixels_removed": max(
                            (v["object_pixels_removed"] for v in views), default=0
                        ),
                        "useful_visible_fraction_panel": float(
                            np.mean([v["useful_visible_occluded"] for v in views])
                        )
                        if views
                        else None,
                        "useful_visible_fraction_panel_hidden": float(
                            np.mean([v["useful_visible_counterfactual"] for v in views])
                        )
                        if views
                        else None,
                    }
                )
    return rows


def compact_report(report):
    """Small report input without per-frame records or any learned-policy claim."""
    episodes = report["episodes"]
    return {
        "purpose": report["purpose"],
        "occlusion_revision": report["occlusion_revision"],
        "backend": "native MuJoCo",
        "training_started": False,
        "learned_policies_evaluated": False,
        "controls": report["controls"],
        "limitations": report["limitations"],
        "useful_visibility_rule": report["useful_visibility_rule"],
        "seeds": report["seeds"],
        "tasks": [
            {
                "task": task,
                "episodes": len(
                    selected := [e for e in episodes if e["experiment"]["task"] == task]
                ),
                "scripted_successes_held_3_steps": sum(
                    e["scripted_success_held_3_steps"] for e in selected
                ),
                "clean_dynamic_physics_identical_every_step": all(
                    e["matched_clean_dynamic_qpos_exactly_equal_every_step"] for e in selected
                ),
                "mean_camera_cartesian_travel_m": float(
                    np.mean([e["camera_cartesian_travel_m"] for e in selected])
                ),
                "configuration": selected[0]["experiment"],
            }
            for task in sorted({e["experiment"]["task"] for e in episodes})
        ],
        "rendered_visibility_summary": report["rendered_visibility_summary"],
        "media": [
            {
                "task": e["experiment"]["task"],
                "seed": e["experiment"]["seed"],
                "video": e["video"],
                "snapshot": e["snapshot"],
            }
            for e in episodes
            if e["video"]
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20000, 20001, 20002])
    parser.add_argument("--output", type=Path, default=Path("artifacts/dynamic_occlusion"))
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--randomize", action="store_true", help="Also randomize manipulation resets"
    )
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("stride must be positive")
    torch.set_num_threads(1)
    episodes = []
    for task in args.tasks:
        for index, seed in enumerate(args.seeds):
            cfg = Experiment(
                task=task,
                occlusion="dynamic",
                num_envs=1,
                seed=seed,
                randomize=args.randomize,
                fixed_position=static_candidates("plug")[7] if task == "plug" else None,
            )
            episode = run_episode(
                cfg, args.output, stride=args.stride, record_video=index == 0 and not args.no_videos
            )
            episodes.append(episode)
            print(
                json.dumps(
                    {
                        "task": task,
                        "seed": seed,
                        "scripted_feasible": episode["scripted_success_held_3_steps"],
                        "optical_only_physics": episode[
                            "matched_clean_dynamic_qpos_exactly_equal_every_step"
                        ],
                    }
                ),
                flush=True,
            )
    rows = summarize(episodes)
    report = {
        "purpose": "Native optical feasibility; not learned task success or an active-policy benchmark",
        "occlusion_revision": "world_sweep_v1",
        "useful_visibility_rule": "At least three prong pixels for plug; object pixels for transfer/push. Geometry evidence only, not pose-estimation accuracy.",
        "controls": "Every render has a same-state panel-hidden counterfactual. A separate clean environment receives identical scripted actions; qpos is compared at every step. Fixed plug view 7 is a diagnostic view, not selected on these episodes.",
        "limitations": "Privileged scripted manipulator and camera controls; native MuJoCo only. Plug camera path is time-only, other scripted camera paths track true object/goal. Immutable hidden geometry can be remembered; waiting is allowed and the panel clears before timeout. A fixed view may avoid the obstruction. Neither motion nor information gain alone establishes task-success improvement.",
        "seeds": args.seeds,
        "episodes": episodes,
        "rendered_visibility_summary": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(compact_report(report), indent=2) + "\n")
    with (args.output / "visibility.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
