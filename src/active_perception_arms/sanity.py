"""Render scripted rollouts and quantify views at the policy's actual resolution."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .config import OCCLUSIONS, TASKS, Experiment
from .native import NativeEnv, ScriptedPolicy


class CameraAudit:
    def __init__(self, env):
        self.env = env
        self.renderer = mujoco.Renderer(env.model, height=env.cfg.height, width=env.cfg.width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = (1, 1, 1, 0, 0, 0)
        self.option.sitegroup[:] = 0
        self.object_geoms = np.where(env.model.geom_bodyid == env.model.body("object/object").id)[0]
        self.fixture_geoms = np.where(
            env.model.geom_bodyid == env.model.body("fixture/fixture").id
        )[0]
        self.table_id = env.model.geom("table/top").id
        self.panel_id = env.model.geom("occluder/panel").id

    def render(self, camera):
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.env.data, camera=camera, scene_option=self.option)
        rgb = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.env.data, camera=camera, scene_option=self.option)
        seg = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        ids = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        metrics = {
            "object_pixels": int(np.isin(ids, self.object_geoms).sum()),
            "fixture_pixels": int(np.isin(ids, self.fixture_geoms).sum()),
            "table_fraction": float((ids == self.table_id).mean()),
            "occluder_pixels": int((ids == self.panel_id).sum()),
        }
        if self.env.cfg.task == "plug":
            metrics["pin_pixels"] = int((ids == self.env.model.geom("object/pin").id).sum())
        cid = self.env.model.camera(camera).id
        local = self.env.data.cam_xmat[cid].reshape(3, 3).T @ (
            self.env.goal - self.env.data.cam_xpos[cid]
        )
        depth = -local[2]
        half_height = depth * np.tan(np.deg2rad(self.env.model.cam_fovy[cid]) / 2)
        metrics["goal_in_frame"] = bool(
            depth > 0
            and abs(local[1]) < half_height
            and abs(local[0]) < half_height * self.env.cfg.width / self.env.cfg.height
        )
        return rgb, metrics

    def close(self):
        self.renderer.close()


def mosaic(images, labels, scale=3):
    h, w = images[0].shape[:2]
    canvas = Image.new("RGB", (w * scale * len(images), h * scale + 42), (20, 24, 30))
    draw = ImageDraw.Draw(canvas)
    for i, (rgb, label) in enumerate(zip(images, labels, strict=True)):
        canvas.paste(
            Image.fromarray(rgb).resize((w * scale, h * scale), Image.Resampling.NEAREST),
            (i * w * scale, 0),
        )
        draw.text((i * w * scale + 4, h * scale + 4), label, fill="white")
    return canvas


def run(task, output, *, occlusion="clean", seed=0, seconds=12.0, randomize=False, video=True):
    cfg = Experiment(
        task=task,
        condition="active",
        occlusion=occlusion,
        seed=seed,
        episode_seconds=seconds,
        randomize=randomize,
        num_envs=1,
    )
    env = NativeEnv(cfg)
    policy, audit = ScriptedPolicy(env), CameraAudit(env)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cams = ("manipulator/wrist_cam", "fixed", "camera_arm/wrist_cam")
    records, snapshots = [], []
    writer = (
        imageio.get_writer(
            output / f"{task}_{occlusion}.mp4", fps=1 / (cfg.step_dt * 3), macro_block_size=2
        )
        if video
        else None
    )
    successful = False
    success_hold = 0
    max_contacts = 0
    try:
        for step in range(int(seconds / cfg.step_dt)):
            success_hold = success_hold + 1 if env.step(policy()) else 0
            successful = success_hold >= 3 or successful
            max_contacts = max(max_contacts, env.data.ncon)
            if not np.isfinite(env.data.qpos).all():
                raise RuntimeError(f"Non-finite physics in {task} at step {step}")
            if step % 3:
                continue
            images, labels = [], []
            rec = {"time": float(env.data.time), "stage": policy.stage, "success": env.success()}
            for name in cams:
                rgb, metrics = audit.render(name)
                images.append(rgb)
                rec[name] = metrics
                labels.append(
                    f"{name}\nobject={metrics['object_pixels']} table={metrics['table_fraction']:.0%}"
                )
            tile = mosaic(images, labels)
            if writer:
                writer.append_data(np.asarray(tile))
            if step % 60 == 0:
                snapshots.append(tile)
            records.append(rec)
        report = {
            "task": task,
            "occlusion": occlusion,
            "seed": seed,
            "scripted_success": successful,
            "final_success": env.success(),
            "final_object": env.object_pos.tolist(),
            "goal": env.goal.tolist(),
            "max_contacts": int(max_contacts),
            "physics": "native MuJoCo",
            "policy_resolution": [cfg.width, cfg.height],
            "frames": records,
        }
        (output / f"{task}_{occlusion}.json").write_text(json.dumps(report, indent=2))
        gallery = Image.new("RGB", (snapshots[0].width, sum(x.height for x in snapshots)))
        y = 0
        for snapshot in snapshots:
            gallery.paste(snapshot, (0, y))
            y += snapshot.height
        gallery.save(output / f"{task}_{occlusion}.png")
        # Independent overview makes table/arm/fixture placement reviewable.
        world = mujoco.Renderer(env.model, height=600, width=800)
        try:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = (0, 0.16, 0.08)
            camera.distance, camera.azimuth, camera.elevation = 1.15, 145, -30
            world.update_scene(env.data, camera=camera, scene_option=audit.option)
            Image.fromarray(world.render()).save(output / f"{task}_overview.png")
        finally:
            world.close()
        return {k: v for k, v in report.items() if k != "frames"}
    finally:
        audit.close()
        if writer:
            writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--occlusion", choices=OCCLUSIONS, default="clean")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sanity"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(1)
    reports = [
        run(
            task,
            args.output,
            occlusion=args.occlusion,
            seed=args.seed,
            seconds=args.seconds,
            randomize=args.randomize,
            video=not args.no_video,
        )
        for task in args.tasks
    ]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
