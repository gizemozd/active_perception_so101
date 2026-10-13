"""Render the hidden_prongs_v2 information structure: fixed vs scripted active camera.

Both rollouts share one reset and the same privileged lift-align-descend
manipulator script. The fixed camera stays at the shared start pose; the
scripted active camera uses the privileged gap position to look through the
screen gap at the plug underside. This illustrates the sensing problem only. It
is not a learned policy and must not be reported as a success rate.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from active_perception_arms import plug, plug_v2
from active_perception_arms.config import Experiment
from active_perception_arms.native import NativeEnv, ScriptedPolicy, native_plug_jammed
from active_perception_arms.robots.camera_control import LOOKAT_BOX, LOOKAT_CENTER, LOOKAT_HALF


def active_camera_action(env):
    """Privileged script: camera on the line plug-underside -> gap centre, low height."""
    low, high = (np.array(v) for v in LOOKAT_BOX["plug"])
    target = env.object_pos.copy()
    target[2] -= 0.006
    gap = np.array([env.screen_gap, plug_v2.SCREEN_Y, target[2]])
    y = 0.20
    scale = (y - target[1]) / (gap[1] - target[1])
    position = target + (gap - target) * scale
    position[1] = y
    position[2] = 0.055
    position = np.clip(position, low, high)
    raw = np.zeros(5)
    raw[:3] = 2 * (position - low) / (high - low) - 1
    raw[3:] = np.clip(
        (target[:2] - np.array(LOOKAT_CENTER["plug"][:2])) / np.array(LOOKAT_HALF["plug"]), -1, 1
    )
    return raw


def label(image, text):
    canvas = Image.fromarray(image)
    ImageDraw.Draw(canvas).rectangle((0, 0, canvas.width, 22), fill=(15, 18, 24))
    ImageDraw.Draw(canvas).text((6, 5), text, fill=(240, 240, 240))
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="xm", choices=plug.VARIANTS)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--output", type=Path, default=Path("artifacts/diagnosis/demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = Experiment(
        task="plug",
        condition="active",
        control_revision="pose_v1",
        task_revision="hidden_prongs_v2",
        plug_screen="gap_v1",
        camera_action="lookat_v2",
        camera_start="lookat_v2",
        plug_variant=args.variant,
        num_envs=1,
    )
    frames, summary = {}, {}
    for mode in ("fixed", "active"):
        env = NativeEnv(cfg)
        env.reset(args.seed)
        policy = ScriptedPolicy(env)
        small = mujoco.Renderer(env.model, 96, 128)
        big = mujoco.Renderer(env.model, 360, 640)
        option = mujoco.MjvOption()
        option.geomgroup[:] = (1, 1, 1, 0, 0, 0)
        option.sitegroup[:] = 0
        prongs = [env.model.geom(f"object/prong_{t}").id for t in "ab"]
        hold, rows, video = 0, [], []
        for k in range(int(cfg.episode_seconds / cfg.step_dt)):
            action = policy()
            action[cfg.manip_dim :] = active_camera_action(env) if mode == "active" else 0.0
            if mode == "fixed":
                # Hold the camera at the shared start pose (the fixed baseline).
                low, high = (np.array(v) for v in LOOKAT_BOX["plug"])
                start = env.camera_pose_ik.initial.numpy()[:3]
                action[cfg.manip_dim : cfg.manip_dim + 3] = 2 * (start - low) / (high - low) - 1
            seated = env.step(action)
            hold = hold + 1 if seated else 0
            small.enable_segmentation_rendering()
            small.update_scene(env.data, camera="camera_arm/wrist_cam", scene_option=option)
            seg = small.render()
            small.disable_segmentation_rendering()
            ids = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
            visible = int(np.isin(ids, prongs).sum())
            small.update_scene(env.data, camera="camera_arm/wrist_cam", scene_option=option)
            small.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            camera = small.render()
            small.update_scene(env.data, camera="manipulator/wrist_cam", scene_option=option)
            small.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            wrist = small.render()
            big.update_scene(env.data, camera="overview")
            overview = big.render()
            t = k * cfg.step_dt
            rows.append({"t": t, "prong_pixels": visible, "seated": bool(seated)})
            up = lambda im: np.asarray(Image.fromarray(im).resize((480, 360), Image.NEAREST))  # noqa: E731
            panel = np.concatenate(
                (
                    label(up(camera), f"{mode} camera-arm view  t={t:.2f}s  prong px={visible}"),
                    label(up(wrist), "manipulator wrist view"),
                    label(np.asarray(Image.fromarray(overview).resize((640, 360))), "overview"),
                ),
                axis=1,
            )
            video.append(panel)
            if hold >= 3 or native_plug_jammed(env):
                break
        imageio.mimwrite(args.output / f"{mode}.mp4", video, fps=25, macro_block_size=2)
        frames[mode] = video
        summary[mode] = {
            "steps": len(rows),
            "seated_held": hold >= 3,
            "steps_with_prongs_visible": sum(r["prong_pixels"] >= 3 for r in rows),
            "screen_gap_m": env.screen_gap,
            "variant": args.variant,
        }
        small.close()
        big.close()
    picks = [10, 30, 45]
    strip = np.concatenate(
        [np.concatenate([frames[m][min(i, len(frames[m]) - 1)] for i in picks], 0) for m in frames],
        1,
    )
    Image.fromarray(strip).save(args.output / "fixed_vs_active.png")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
