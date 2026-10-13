"""Reset-state geometric information opportunity under the shared camera bounds.

This is a segmentation oracle, not a trained policy or a task-success test.
Endpoints are reached by pose increments with the same iterative IK as the policy.
"""

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from active_perception_arms.config import Experiment
from active_perception_arms.native import NativeEnv, occluder_position
from active_perception_arms.robots.kinematics import axis_to_pan_tilt, pan_tilt_to_axis
from active_perception_arms.sanity import CameraAudit


def useful(metrics, task):
    names = (
        ("prong_a_pixels", "prong_b_pixels", "fixture_pixels")
        if task == "plug"
        else ("object_pixels", "fixture_pixels")
    )
    return all(metrics[k] >= 3 for k in names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", choices=("local_v1", "inspection_v2"), default="local_v1")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Probe five panel sweep phases with the manipulation scene held at reset",
    )
    parser.add_argument(
        "--realized",
        action="store_true",
        help="Recalibrate useful near-limit views to their attained FK poses",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path("artifacts/common_control_study/view_audit") / (
        args.workspace
        + ("_realized" if args.realized else "")
        + ("_dynamic" if args.dynamic else "")
    )
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Native reset-state segmentation at policy resolution. Collision-checked kinematic paths, not dynamic task completion. Visibility is an information proxy, not proof of usable target estimation.",
        "tasks": {},
        "panel_phases": [0, 0.25, 0.5, 0.75, 1] if args.dynamic else None,
    }
    offsets = [np.zeros(5)]
    for i in range(5):
        for sign in (-1, 1):
            x = np.zeros(5)
            x[i] = sign
            offsets.append(x)
    for x in (-1, 1):
        for y in (-1, 1):
            offsets.append(np.array([x, y, -1, 0, 0]))
    for task in ("plug", "push"):
        cfg = Experiment(
            task=task,
            condition="active",
            control_revision="pose_v1",
            camera_workspace=args.workspace,
            num_envs=1,
            width=128,
            height=96,
            occlusion="dynamic" if args.dynamic else "clean",
        )
        if task == "push":
            cfg = replace(
                cfg,
                task_revision="push_v2",
                push_target_layout="arc_v2",
                push_visibility_revision="crossbar_v3",
                push_workspace_revision="arc_clear_v2",
                reset_xy_jitter=0.02,
                reward_profile="push_dense",
            )
        rows = []
        for variant in ("xm", "xp", "ym", "yp") if task == "plug" else (None,):
            env = NativeEnv(replace(cfg, plug_variant=variant))
            audit = CameraAudit(env)
            ik = env.camera_pose_ik
            camera_geoms = {
                i
                for i in range(env.model.ngeom)
                if env.model.geom(i).name.startswith("camera_arm/")
            }
            paths = []
            targets = [ik.home + ik.tensor(offset) * (ik.high - ik.home) for offset in offsets]
            if args.workspace == "inspection_v2":
                targets = [ik.home.clone()]
                for x in (-0.14, 0, 0.14):
                    for y in (-0.07, 0, 0.07):
                        for z in (-0.14, -0.07, 0):
                            position = ik.home[:3] + ik.tensor([x, y, z])
                            pan, tilt = axis_to_pan_tilt(ik.tensor(cfg.fixed_lookat) - position)
                            targets.append(torch.cat([position, pan[None], tilt[None]]))
            for target in targets:
                q = ik.tensor(env.home["camera_arm"][:5])[None]
                path = []
                for alpha in np.linspace(0, 1, 33):
                    q = ik.solve((ik.home + alpha * (target - ik.home))[None], q)
                    path.append(q[0].numpy().copy())
                paths.append((target, path))
            if args.realized:
                if args.workspace != "inspection_v2":
                    raise ValueError("Realized endpoint calibration requires inspection_v2")
                selected = (0, 13, 22, 25) if task == "plug" else (0, 4, 13, 18)
                calibrated = []
                for index in selected:
                    env.data.qpos[env.qadr["camera_arm"][:5]] = paths[index][1][-1]
                    mujoco.mj_forward(env.model, env.data)
                    cid = env.model.camera("camera_arm/wrist_cam").id
                    position = ik.tensor(env.data.cam_xpos[cid].copy())
                    axis = ik.tensor(-env.data.cam_xmat[cid].reshape(3, 3)[:, 2].copy())
                    pan, tilt = axis_to_pan_tilt(axis)
                    target = torch.cat([position, pan[None], tilt[None]])
                    q = ik.tensor(env.home["camera_arm"][:5])[None]
                    path = []
                    for alpha in np.linspace(0, 1, 33):
                        q = ik.solve((ik.home + alpha * (target - ik.home))[None], q)
                        path.append(q[0].numpy().copy())
                    calibrated.append((target, path))
                paths = calibrated
                targets = [t for t, _ in paths]
            samples = itertools.product(
                range(130000, 130000 + (8 if task == "plug" else 32)),
                (0, 0.25, 0.5, 0.75, 1) if args.dynamic else (None,),
            )
            for seed, panel_phase in samples:
                env.reset(seed)
                if panel_phase is not None:
                    env.data.mocap_pos[env.panel_id] = occluder_position(
                        env.onset + panel_phase * env.duration,
                        env.onset,
                        env.duration,
                        env.side,
                        cfg,
                        center=env.panel_center,
                        travel=env.panel_travel,
                    )
                    mujoco.mj_forward(env.model, env.data)
                _, wrist = audit.render("manipulator/wrist_cam")
                _, home_image = audit.render("camera_arm/wrist_cam")
                home_pair = useful(
                    {
                        k: max(v, wrist.get(k, 0))
                        for k, v in home_image.items()
                        if k.endswith("pixels")
                    },
                    task,
                )
                views = []
                for target, path in paths:
                    contacts = set()
                    for q in path:
                        env.data.qpos[env.qadr["camera_arm"][:5]] = q
                        mujoco.mj_forward(env.model, env.data)
                        for c in env.data.contact[: env.data.ncon]:
                            a, b = int(c.geom1), int(c.geom2)
                            if (a in camera_geoms) != (b in camera_geoms):
                                other = b if a in camera_geoms else a
                                if env.model.geom_contype[other]:
                                    contacts.add(env.model.geom(other).name)
                    cid = env.model.camera("camera_arm/wrist_cam").id
                    actual = env.data.cam_xpos[cid]
                    axis = -env.data.cam_xmat[cid].reshape(3, 3)[:, 2]
                    expected = pan_tilt_to_axis(target[3], target[4]).numpy()
                    pos_error = float(np.linalg.norm(actual - target[:3].numpy()))
                    angle = float(np.arccos(np.clip(axis @ expected, -1, 1)))
                    _, metrics = audit.render("camera_arm/wrist_cam")
                    _, current_wrist = audit.render("manipulator/wrist_cam")
                    combined = {
                        k: max(v, current_wrist.get(k, 0))
                        for k, v in metrics.items()
                        if k.endswith("pixels")
                    }
                    in_bounds = bool(((target >= ik.low) & (target <= ik.high)).all())
                    valid = in_bounds and not contacts and pos_error < 0.005 and angle < 0.1
                    views.append(
                        dict(
                            valid=valid,
                            visible_pair=useful(combined, task),
                            useful_pair=valid and useful(combined, task),
                            pixels=metrics,
                            position_error_m=pos_error,
                            axis_error_rad=angle,
                            contacts=sorted(contacts),
                        )
                    )
                rows.append(
                    dict(
                        seed=seed,
                        variant=variant,
                        panel_phase=panel_phase,
                        wrist=wrist,
                        home_pair=home_pair,
                        views=views,
                    )
                )
            audit.renderer.close()
        counts = [sum(r["views"][i]["useful_pair"] for r in rows) for i in range(len(targets))]
        summary = dict(
            episodes=len(rows),
            home_pair=sum(r["home_pair"] for r in rows),
            best_single_endpoint=max(counts),
            best_single_endpoint_index=int(np.argmax(counts)),
            per_reset_oracle=sum(any(v["useful_pair"] for v in r["views"]) for r in rows),
            valid_endpoints_per_reset=[sum(v["valid"] for v in r["views"]) for r in rows],
            endpoint_counts=counts,
        )
        report["tasks"][task] = dict(
            summary=summary, rows=rows, targets=[t.tolist() for t in targets]
        )
        print(task, summary, flush=True)
    report["normalized_offsets"] = [x.tolist() for x in offsets]
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
