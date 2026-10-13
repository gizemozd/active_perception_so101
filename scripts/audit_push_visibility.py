"""Rendered reset-state audit for the fixed-FOV push crossbar experiment."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from active_perception_arms import scenes
from active_perception_arms.config import CAMERA_HOME, FOVY, Experiment
from active_perception_arms.native import NativeEnv
from active_perception_arms.sanity import CameraAudit, mosaic


def pair(m):
    return m["object_pixels"] >= 3 and m["fixture_pixels"] >= 3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=64)
    args = p.parse_args()
    torch.set_num_threads(1)
    cfg = Experiment(
        task="push",
        condition="wrist_static",
        task_revision="push_v2",
        reward_profile="push_dense",
        manip_control="cartesian",
        occlusion="clean",
        reset_xy_jitter=0.02,
        num_envs=1,
        push_visibility_revision="crossbar_v3",
    )
    env = NativeEnv(cfg)
    audit = CameraAudit(env)
    fixed = env.model.camera("fixed")
    # Endpoints on the far side and above the crossbar. IK and forward kinematics
    # determine whether these are actual camera-arm poses; no FOV is altered.
    requests = []
    for vi, pos in enumerate(
        (
            (-0.12, -0.19, 0.27),
            (-0.18, -0.14, 0.22),
            (-0.22, -0.08, 0.18),
            (-0.16, -0.10, 0.24),
            (-0.10, -0.16, 0.20),
            (-0.20, -0.18, 0.20),
        )
    ):
        for tag, target in (("source", (-0.025, -0.045, 0.009)), ("goal", (-0.025, 0.055, 0.009))):
            requests.append((f"view_{vi}_{tag}", pos, target))
    endpoints = []
    for name, pos, target in requests:
        try:
            q = env.ik.camera(np.array(pos), np.array(target), env.home["camera_arm"][:5])
            env.data.qpos[env.qadr["camera_arm"][:5]] = q
            mujoco.mj_forward(env.model, env.data)
            cid = env.model.camera("camera_arm/wrist_cam").id
            actual = env.data.cam_xpos[cid].copy()
            rot = env.data.cam_xmat[cid].reshape(3, 3)
            direction = np.asarray(target) - actual
            direction /= np.linalg.norm(direction)
            angle = float(np.degrees(np.arccos(np.clip(direction @ -rot[:, 2], -1, 1))))
            err = float(np.linalg.norm(actual - pos))
            endpoints.append(
                dict(
                    name=name,
                    requested=pos,
                    target=target,
                    q=q.tolist(),
                    actual=actual.tolist(),
                    position_error=err,
                    angle_error_deg=angle,
                    reachable=err < 0.035 and angle < 10,
                )
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            endpoints.append(
                dict(name=name, requested=pos, target=target, reachable=False, error=repr(exc))
            )

    rows, gallery = [], []
    fixed_fovy = float(env.model.cam_fovy[fixed.id])
    wrist_id = env.model.camera("manipulator/wrist_cam").id
    wrist_fovy = float(env.model.cam_fovy[wrist_id])
    cam_fovy = float(env.model.cam_fovy[env.model.camera("camera_arm/wrist_cam").id])
    crossbar_geom = env.model.geom("visibility/crossbar").id
    camera_body_geoms = np.flatnonzero(
        np.char.startswith(
            np.array([env.model.geom(i).name for i in range(env.model.ngeom)]), "camera_arm/"
        )
    )
    for endpoint in endpoints:
        if not endpoint["reachable"]:
            continue
        q0, q1 = env.home["camera_arm"][:5], np.asarray(endpoint["q"])
        collisions = []
        for alpha in np.linspace(0, 1, 21):
            env.data.qpos[env.qadr["camera_arm"][:5]] = q0 * (1 - alpha) + q1 * alpha
            mujoco.mj_forward(env.model, env.data)
            mujoco.mj_collision(env.model, env.data)
            for ci in range(env.data.ncon):
                contact = env.data.contact[ci]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if g1 in camera_body_geoms or g2 in camera_body_geoms:
                    other = g2 if g1 in camera_body_geoms else g1
                    if (
                        other not in camera_body_geoms
                        and other != crossbar_geom
                        and env.model.geom_contype[other] != 0
                    ):
                        collisions.append(
                            dict(alpha=float(alpha), other=env.model.geom(other).name)
                        )
        endpoint["path_clear"] = not collisions
        endpoint["path_collision_samples"] = collisions[:12]
    try:
        for seed in range(40000, 40000 + args.seeds):
            env.reset(seed)
            _, wrist = audit.render("manipulator/wrist_cam")
            fixed.pos = CAMERA_HOME
            fixed.quat = scenes.look_at_quat(CAMERA_HOME, cfg.fixed_lookat)
            mujoco.mj_forward(env.model, env.data)
            _, static = audit.render("fixed")
            row = dict(seed=seed, wrist=wrist, static=static, endpoints={})
            for endpoint in endpoints:
                if not endpoint["reachable"] or not endpoint.get("path_clear", False):
                    continue
                env.data.qpos[env.qadr["camera_arm"][:5]] = endpoint["q"]
                mujoco.mj_forward(env.model, env.data)
                _, metrics = audit.render("camera_arm/wrist_cam")
                row["endpoints"][endpoint["name"]] = metrics
            rows.append(row)
            if seed == 40000:
                images, labels = [], []
                env.reset(seed)
                rgb, _ = audit.render("manipulator/wrist_cam")
                images.append(rgb)
                labels.append("wrist")
                fixed.pos = CAMERA_HOME
                fixed.quat = scenes.look_at_quat(CAMERA_HOME, cfg.fixed_lookat)
                mujoco.mj_forward(env.model, env.data)
                rgb, _ = audit.render("fixed")
                images.append(rgb)
                labels.append("static home")
                for endpoint in endpoints:
                    if endpoint["reachable"] and endpoint.get("path_clear", False):
                        env.data.qpos[env.qadr["camera_arm"][:5]] = endpoint["q"]
                        mujoco.mj_forward(env.model, env.data)
                        rgb, _ = audit.render("camera_arm/wrist_cam")
                        images.append(rgb)
                        labels.append(endpoint["name"])
                gallery.append(mosaic(images, labels, scale=3))

        def rate(fn):
            return float(np.mean([fn(r) for r in rows]))

        summary = {
            "episodes": len(rows),
            "wrist_pair_rate": rate(lambda r: pair(r["wrist"])),
            "static_pair_rate": rate(lambda r: pair(r["static"])),
            "wrist_plus_static_pair_rate": rate(
                lambda r: (
                    any(max(r["wrist"][k], r["static"][k]) >= 3 for k in ("object_pixels",))
                    and any(max(r["wrist"][k], r["static"][k]) >= 3 for k in ("fixture_pixels",))
                )
            ),
            "active_single_view_pair_rate": rate(
                lambda r: any(pair(m) for m in r["endpoints"].values())
            ),
            "active_scan_pair_rate": rate(
                lambda r: (
                    any(m["object_pixels"] >= 3 for m in r["endpoints"].values())
                    and any(m["fixture_pixels"] >= 3 for m in r["endpoints"].values())
                )
            ),
            "active_candidate_rates": {
                ep["name"]: float(
                    np.mean(
                        [
                            pair(r["endpoints"][ep["name"]])
                            for r in rows
                            if ep["name"] in r["endpoints"]
                        ]
                    )
                )
                for ep in endpoints
                if ep["reachable"]
            },
            "fovy": {
                "wrist": wrist_fovy,
                "static": fixed_fovy,
                "active": cam_fovy,
                "configured": FOVY,
            },
            "active_path_clear_endpoint_count": sum(bool(e.get("path_clear")) for e in endpoints),
        }
        report = dict(
            experiment=cfg.to_dict(),
            summary=summary,
            endpoints=endpoints,
            rows=rows,
            criterion="Visible means >=3 segmentation pixels for source body and goal marker.",
            scope="Reset-state visibility only; reachability is IK pose tolerance, path clearance still required.",
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "visibility-audit.json").write_text(json.dumps(report, indent=2) + "\n")
        if gallery:
            gallery[0].save(args.output / "visibility-gallery.png")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        audit.close()


if __name__ == "__main__":
    main()
