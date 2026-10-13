#!/usr/bin/env python3
"""Screen transfer cameras at policy resolution along successful physical rollouts."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from active_perception_arms.config import Experiment, static_candidates
from active_perception_arms.native import NativeEnv, ScriptedPolicy
from active_perception_arms.sanity import CameraAudit, mosaic
from active_perception_arms.scenes import look_at_quat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows, images = [], {}
    grid = static_candidates("transfer")
    for seed in (0, 10000, 10015):
        cfg = Experiment(
            task="transfer",
            condition="wrist_static",
            occlusion="clean",
            num_envs=1,
            randomize=seed != 0,
            seed=seed,
        )
        env = NativeEnv(cfg)
        controller, renderer = ScriptedPolicy(env), CameraAudit(env)
        sampled = set()
        try:
            for step in range(145):
                # First state in every control stage; stage 7 also samples settled release.
                key = "settled" if step == 144 else str(controller.stage)
                if key not in sampled:
                    sampled.add(key)
                    for name in ("manipulator/wrist_cam", "camera_arm/wrist_cam"):
                        rgb, metrics = renderer.render(name)
                        rows.append(dict(seed=seed, step=step, phase=key, camera=name, **metrics))
                        if seed == 0:
                            images[(key, name)] = rgb
                    for index, position in enumerate(grid):
                        camera = env.model.camera("fixed")
                        camera.pos = position
                        camera.quat = look_at_quat(position, cfg.fixed_lookat)
                        mujoco.mj_forward(env.model, env.data)
                        rgb, metrics = renderer.render("fixed")
                        name = f"fixed_{index:02d}"
                        rows.append(dict(seed=seed, step=step, phase=key, camera=name, **metrics))
                        if seed == 0:
                            images[(key, name)] = rgb
                env.step(controller())
        finally:
            renderer.close()
    summary = []
    for camera in sorted({r["camera"] for r in rows}):
        group = [r for r in rows if r["camera"] == camera]
        summary.append(
            dict(
                camera=camera,
                samples=len(group),
                min_object_pixels=min(r["object_pixels"] for r in group),
                mean_object_pixels=float(np.mean([r["object_pixels"] for r in group])),
                useful_object_fraction=float(np.mean([r["object_pixels"] >= 3 for r in group])),
                goal_in_frame_fraction=float(np.mean([r["goal_in_frame"] for r in group])),
            )
        )
    fixed = [r for r in summary if r["camera"].startswith("fixed_")]
    best = max(
        fixed,
        key=lambda r: (
            r["useful_object_fraction"],
            r["min_object_pixels"],
            r["mean_object_pixels"],
        ),
    )["camera"]
    selected = ["manipulator/wrist_cam", "camera_arm/wrist_cam", "fixed_00", best]
    tiles = []
    for phase in ("0", "2", "4", "6", "settled"):
        tiles.append(
            mosaic(
                [images[(phase, camera)] for camera in selected],
                [f"phase {phase}: {camera}" for camera in selected],
                scale=3,
            )
        )
    gallery = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)))
    for index, tile in enumerate(tiles):
        gallery.paste(tile, (0, index * tile.height))
    gallery.save(args.output / "views.png")
    report = dict(
        task_revision=cfg.task_revision,
        resolution=[cfg.width, cfg.height],
        scope="Native segmentation along three privileged clean rollouts; visibility only, not policy success or trained view selection.",
        useful_threshold_pixels=3,
        fixed_grid=grid,
        gallery_fixed_view=best,
        rows=rows,
        summary=summary,
    )
    (args.output / "views.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
