"""Render nine native resets from the frozen push RGB study distribution."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PLAN = json.loads((ROOT / "artifacts/push_rgb/study-plan.json").read_text())
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from active_perception_arms.config import Experiment
from active_perception_arms.native import NativeEnv

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
cfg = Experiment(task="push", condition="wrist_static", task_revision="push_v2",
                 reward_profile="push_dense", manip_control="cartesian",
                 occlusion="clean", reset_xy_jitter=0.02, num_envs=1,
                 push_visibility_revision="crossbar_v3")
env = NativeEnv(cfg)
# Only the diagnostic observer camera changes. Policy cameras and scene are intact.
cid = env.model.camera("overview").id
env.model.cam_pos[cid] = (-0.025, -0.005, 0.65)
env.model.cam_quat[cid] = (1, 0, 0, 0)
env.model.cam_fovy[cid] = 25
width, height, caption, gap, header = 500, 420, 68, 14, 116
canvas = Image.new("RGB", (3 * width + 4 * gap, header + 3 * (height + caption + gap) + gap), "#101923")
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=19)
small = ImageFont.load_default(size=16)
title = ImageFont.load_default(size=30)
draw.text((gap, 15), "Push task: targets sampled around the block", font=title, fill="white")
draw.text((gap, 57), "Overhead diagnostic view; crossbar hidden | Green = target marker | Orange = push object", font=font, fill="#c8d6e4")
draw.text((gap, 86), "Target: -80 to +80 deg, radius 50-120 mm | Green marker: 20 x 20 mm | Success: center within 5 mm", font=small, fill="#c8d6e4")
option = mujoco.MjvOption()
option.geomgroup[:] = (1, 1, 1, 0, 0, 0)
option.sitegroup[:] = 0
# Hide only the optical crossbar in this observer illustration so every target is readable.
env.model.geom("visibility/crossbar").group = 5
rows = []
renderer = mujoco.Renderer(env.model, height=height, width=width)
try:
    for i, seed in enumerate(range(40000, 40009)):
        env.reset(seed)
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, camera="overview", scene_option=option)
        frame = Image.fromarray(renderer.render().copy())
        frame.save(OUT / f"reset-{seed}.png")
        x = gap + (i % 3) * (width + gap)
        y = header + (i // 3) * (height + caption + gap)
        canvas.paste(frame, (x, y))
        goal = env.goal.copy()
        delta = (goal[:2] - env.object_pos[:2]) * 1000
        angle = np.degrees(np.arctan2(delta[0], delta[1]))
        radius = np.linalg.norm(delta)
        draw.text((x + 10, y + height + 7), f"{i+1:02d}  |  seed {seed}  |  angle {angle:+.1f} deg | radius {radius:.1f} mm", font=small, fill="white")
        draw.text((x + 10, y + height + 34), f"Target center: x={goal[0]*1000:.1f}, y={goal[1]*1000:.1f} mm", font=small, fill="#aabfd1")
        rows.append(dict(seed=seed, target_xyz_m=goal.tolist(), object_xyz_m=env.object_pos.copy().tolist()))
finally:
    renderer.close()
canvas.save(OUT / "target-initial-conditions-grid.png")
(OUT / "initial-conditions.json").write_text(json.dumps(dict(experiment=cfg.to_dict(), source=str(ROOT / "src"), samples=rows, camera="overhead observer with optical crossbar hidden; not a policy input", sampling="Native NumPy resets from the same uniform distribution; not bitwise Warp reset seeds"), indent=2) + "\n")
print(OUT / "target-initial-conditions-grid.png")
