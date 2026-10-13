"""Matched-variant information audit for the plug task with and without the gap screen.

For each sampled reset, the identical scene is rendered with all four hidden
prong variants. A viewpoint is *informative* when every variant pair differs in
at least ``--min-pixels`` RGB pixels at policy resolution (128x96, 48.46 deg).
This is an information-opportunity measurement, not a learned-policy result.

Candidate cameras:
* manipulator wrist camera;
* camera-arm poses reachable while looking at the workspace (the pose_v1
  "fixed" baseline is the best single such pose; the active oracle may pick the
  best pose per episode);
* the 26 historical free-standing fixed views (not reachable by the arm).
"""

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from active_perception_arms import plug, plug_v2
from active_perception_arms.config import FOVY, STOW, Experiment, static_candidates
from active_perception_arms.native import calibration
from active_perception_arms.robots.plug_control import PlugIK
from active_perception_arms.scenes import grasp_relpose, native_spec

LOOKAT = np.array([0.02, 0.03, 0.03])


def camera_grid(ik, screen, coarse=False):
    ys = (0.18, 0.22, 0.26, 0.30) if screen else (0.10, 0.14, 0.18, 0.22, 0.26, 0.30)
    xs = np.arange(-0.10, 0.121, 0.02)
    zs = (0.05, 0.08, 0.11, 0.15, 0.20, 0.26, 0.32)
    if coarse:
        ys = (0.18, 0.24, 0.30) if screen else (0.12, 0.18, 0.24, 0.30)
        xs = np.arange(-0.09, 0.121, 0.03)
        zs = (0.05, 0.08, 0.12, 0.18, 0.26)
    poses = [np.array(p) for p in itertools.product(xs, ys, zs)]
    poses = np.array(poses)
    axis = LOOKAT - poses
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    pos_b = ik.camera_frame.pos_to_base(torch.as_tensor(poses))
    ax_b = ik.camera_frame.vec_to_base(torch.as_tensor(axis))
    best_q = best_err = best_ang = None
    for s in range(5):
        torch.manual_seed(s)
        q0 = (
            torch.as_tensor(np.array(STOW[:5]))[None].expand(len(poses), 5).clone()
            if s == 0
            else ik.camera.joint_low
            + torch.rand(len(poses), 5, dtype=torch.float64)
            * (ik.camera.joint_high - ik.camera.joint_low)
        )
        q = ik.camera.solve_ik(pos_b, q0, target_axis=ax_b, iters=150, damping=0.02, max_dq=0.3)
        p, r = ik.camera.fk(q)
        err = (p - pos_b).norm(dim=-1).numpy()
        a = r @ torch.tensor([0, 0, -1.0], dtype=torch.float64)
        ang = torch.arccos((a * ax_b).sum(-1).clamp(-1, 1)).numpy()
        if best_q is None:
            best_q, best_err, best_ang = q.numpy(), err, ang
        else:
            better = err + 0.1 * ang < best_err + 0.1 * best_ang
            best_q = np.where(better[:, None], q.numpy(), best_q)
            best_err = np.where(better, err, best_err)
            best_ang = np.where(better, ang, best_ang)
    q, err, ang = best_q, best_err, best_ang
    ok = (err < 0.003) & (ang < 0.03)
    return poses[ok], q[ok]


def matched_difference(renderers, datas, option, camera, min_pixels, keep=False):
    """Render one state with every variant; informative if all pairs differ enough."""
    frames = []
    for v in plug.VARIANTS:
        r = renderers[v]
        r.update_scene(datas[v], camera=camera, scene_option=option)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        frames.append(r.render().astype(int))
    diff = min(int((np.abs(a - b).max(-1) > 8).sum()) for a, b in itertools.combinations(frames, 2))
    return diff >= min_pixels, frames if keep else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--min-pixels", type=int, default=6)
    parser.add_argument("--coarse", action="store_true", help="Smaller camera-arm grid")
    parser.add_argument("--scenes", nargs="+", default=["v1_clean", "v2_gap_screen"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/diagnosis/views"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"min_pixels": args.min_pixels, "episodes": args.episodes, "scenes": {}}
    for scene in args.scenes:
        screen = scene == "v2_gap_screen"
        base = Experiment(
            task="plug",
            condition="active",
            control_revision="pose_v1",
            task_revision="hidden_prongs_v2" if screen else "hidden_prongs_v1",
            plug_screen="gap_v1" if screen else "none",
            num_envs=1,
        )
        models = {v: native_spec(replace(base, plug_variant=v)).compile() for v in plug.VARIANTS}
        for m in models.values():
            m.vis.global_.fovy = FOVY
        datas = {v: mujoco.MjData(m) for v, m in models.items()}
        model = models["xm"]
        ik = PlugIK(model, dtype=torch.float64)
        home = calibration("plug")
        poses, arm_q = camera_grid(ik, screen, args.coarse)
        print(scene, "reachable arm poses", len(poses), flush=True)
        renderers = {v: mujoco.Renderer(models[v], height=96, width=128) for v in plug.VARIANTS}
        option = mujoco.MjvOption()
        option.geomgroup[:] = (1, 1, 1, 0, 0, 0)
        option.sitegroup[:] = 0
        free = [np.array(p) for p in static_candidates("plug")]
        free_cams = []
        for p in free:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            free_cams.append((p, cam))
        qadr = {
            arm: np.array(
                [
                    model.joint(f"{arm}/{j}").qposadr[0]
                    for j in (
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                        "gripper",
                    )
                ]
            )
            for arm in ("manipulator", "camera_arm")
        }
        obj_adr = int(model.joint("object/free").qposadr[0])
        fixture = int(model.body("fixture/fixture").mocapid[0])
        screens = (
            [int(model.body(f"screen_{s}/panel").mocapid[0]) for s in ("left", "right")]
            if screen
            else []
        )
        gripper = model.body("manipulator/gripper").id
        p_rel, q_rel = grasp_relpose()
        rng = np.random.default_rng(args.seed)
        arm_hits = np.zeros((args.episodes, len(poses)), dtype=bool)
        free_hits = np.zeros((args.episodes, len(free)), dtype=bool)
        wrist_hits = np.zeros(args.episodes, dtype=bool)
        socket_vis = np.zeros((args.episodes, len(poses)), dtype=bool)
        examples = []
        for e in range(args.episodes):
            if e % 4 == 0:
                print(scene, "episode", e, flush=True)
            xy = np.array(plug.SPAWN_CENTER) + rng.uniform(-1, 1, 2) * np.array(plug.SPAWN_HALF)
            tcp = np.r_[xy, plug.SPAWN_Z] + np.array(plug.GRIP_OFFSET)
            q = ik.reset_manip(torch.as_tensor(tcp)[None])[0].numpy()
            socket = rng.uniform(0, 0.05, 2)
            gap = float(
                plug_v2.sample_gap(torch.tensor([rng.uniform()]), torch.tensor([xy]))[0]
            )
            for v in plug.VARIANTS:
                m, d = models[v], datas[v]
                mujoco.mj_resetData(m, d)
                d.qpos[qadr["manipulator"]] = np.r_[q, 0.27]
                d.qpos[qadr["camera_arm"]] = home["camera_arm"]
                d.mocap_pos[fixture] = (*socket, 0)
                if screen:
                    left, right = plug_v2.screen_segment_centers(torch.tensor([gap]))
                    d.mocap_pos[screens[0]] = left[0].numpy()
                    d.mocap_pos[screens[1]] = right[0].numpy()
                mujoco.mj_kinematics(m, d)
                rot = d.xmat[gripper].reshape(3, 3)
                quat = np.empty(4)
                mujoco.mju_mulQuat(quat, d.xquat[gripper], q_rel)
                d.qpos[obj_adr : obj_adr + 7] = np.r_[d.xpos[gripper] + rot @ p_rel, quat]
                mujoco.mj_forward(m, d)

            wrist_hits[e] = matched_difference(
                renderers, datas, option, "manipulator/wrist_cam", args.min_pixels
            )[0]
            for k, (pos, cam) in enumerate(free_cams):
                axis = LOOKAT - pos
                dist = np.linalg.norm(axis)
                cam.lookat[:] = LOOKAT
                cam.distance = dist
                cam.azimuth = np.degrees(np.arctan2(axis[1], axis[0]))
                cam.elevation = np.degrees(np.arcsin(axis[2] / dist))
                free_hits[e, k] = matched_difference(
                    renderers, datas, option, cam, args.min_pixels
                )[0]
            for k in range(len(poses)):
                for v in plug.VARIANTS:
                    datas[v].qpos[qadr["camera_arm"][:5]] = arm_q[k]
                    mujoco.mj_forward(models[v], datas[v])
                hit, frames = matched_difference(
                    renderers, datas, option, "camera_arm/wrist_cam", args.min_pixels, e < 3
                )
                arm_hits[e, k] = hit
                r = renderers["xm"]
                r.enable_segmentation_rendering()
                r.update_scene(datas["xm"], camera="camera_arm/wrist_cam", scene_option=option)
                seg = r.render()
                r.disable_segmentation_rendering()
                ids = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
                fixture_geoms = np.where(model.geom_bodyid == model.body("fixture/fixture").id)[0]
                socket_vis[e, k] = np.isin(ids, fixture_geoms).sum() >= 50
                if e < 3 and hit and len(examples) < 12 and k % 7 == 0:
                    examples.append(np.concatenate([f.astype(np.uint8) for f in frames], 1))
        fixed_rate = arm_hits.mean(0)
        best_fixed = int(np.argmax(fixed_rate + 1e-3 * socket_vis.mean(0)))
        row = {
            "reachable_arm_poses": len(poses),
            "wrist_informative": float(wrist_hits.mean()),
            "arm_best_single_pose": poses[best_fixed].tolist(),
            "arm_best_single_pose_informative": float(fixed_rate[best_fixed]),
            "arm_best_single_pose_socket_visible": float(socket_vis[:, best_fixed].mean()),
            "arm_oracle_per_episode_informative": float(arm_hits.any(1).mean()),
            "arm_oracle_informative_and_socket": float((arm_hits & socket_vis).any(1).mean()),
            "arm_pose_informative_rate_quantiles": np.quantile(fixed_rate, [0.5, 0.9, 1]).tolist(),
            "free_fixed_best_view": int(np.argmax(free_hits.mean(0))),
            "free_fixed_best_informative": float(free_hits.mean(0).max()),
            "free_fixed_view7_informative": float(free_hits[:, 7].mean()),
            "free_fixed_oracle_informative": float(free_hits.any(1).mean()),
            "informative_poses_per_episode_median": float(np.median(arm_hits.sum(1))),
        }
        report["scenes"][scene] = row
        print(scene, json.dumps(row, indent=1))
        np.savez_compressed(
            args.output / f"{scene}_hits.npz",
            arm=arm_hits,
            free=free_hits,
            wrist=wrist_hits,
            poses=poses,
            socket=socket_vis,
        )
        if examples:
            Image.fromarray(np.concatenate(examples, 0)).save(args.output / f"{scene}_examples.png")
        for r in renderers.values():
            r.close()
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
