"""Native rendered push visibility prototypes; no policy training or claims of success."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from active_perception_arms import native, scenes
from active_perception_arms.config import CAMERA_HOME, Experiment, static_candidates
from active_perception_arms.sanity import CameraAudit, mosaic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=32)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    original = native.native_spec

    def prototype(cfg, **kwargs):
        spec = original(cfg, **kwargs)
        for name, y in (("source", -0.045), ("goal", 0.055)):
            body = spec.worldbody.add_body(name="cover_" + name, mocap=True, pos=(-0.025, y, 0))
            # Open on local south side. This prototype is an optical intervention;
            # manipulation physics and reset coordinates remain identical.
            for label, pos, size in (
                ("roof", (0, 0, 0.12), (0.058, 0.058, 0.003)),
                ("back", (0, 0.058, 0.075), (0.061, 0.003, 0.045)),
                ("left", (-0.058, 0, 0.075), (0.003, 0.058, 0.045)),
                ("right", (0.058, 0, 0.075), (0.003, 0.058, 0.045)),
            ):
                scenes.box(
                    body, name + "_" + label, pos, size, (0.4, 0.45, 0.5, 1), collision=False
                )
        return spec

    native.native_spec = prototype
    cfg = Experiment(
        task="push",
        task_revision="push_v2",
        reward_profile="push_dense",
        manip_control="cartesian",
        reset_xy_jitter=0.02,
        num_envs=1,
        condition="wrist_static",
        occlusion="clean",
    )
    env = native.NativeEnv(cfg)
    audit = CameraAudit(env)
    cover_ids = [int(env.model.body("cover_" + name).mocapid[0]) for name in ("source", "goal")]
    fixed = env.model.camera("fixed")
    cameras = [
        dict(name=f"fixed_{i:02}", position=tuple(pos), lookat=(0, 0.025, 0.025))
        for i, pos in enumerate(static_candidates("push"))
    ]
    reachable = []
    for x, y, z in (
        (-0.12, -0.19, 0.27),
        (-0.18, -0.14, 0.22),
        (-0.22, -0.08, 0.18),
        (-0.20, -0.02, 0.18),
        (-0.16, -0.10, 0.24),
        (-0.10, -0.16, 0.20),
        (-0.20, -0.18, 0.20),
    ):
        for tag, target in (("source", (-0.025, -0.045, 0.009)), ("goal", (-0.025, 0.055, 0.009))):
            q = env.ik.camera(np.array((x, y, z)), target, env.home["camera_arm"][:5])
            env.data.qpos[env.qadr["camera_arm"][:5]] = q
            mujoco.mj_forward(env.model, env.data)
            cam = env.model.camera("camera_arm/wrist_cam").id
            pos = env.data.cam_xpos[cam].copy()
            rot = env.data.cam_xmat[cam].reshape(3, 3).copy()
            desired = np.array(target) - pos
            desired /= np.linalg.norm(desired)
            error = float(np.linalg.norm(pos - np.array((x, y, z))))
            angle = float(np.degrees(np.arccos(np.clip(desired @ -rot[:, 2], -1, 1))))
            if error < 0.035 and angle < 10:
                reachable.append(
                    dict(
                        name=f"reachable_{len(reachable):02}_{tag}",
                        q=q.tolist(),
                        position=pos.tolist(),
                        lookat=(pos - rot[:, 2]).tolist(),
                        requested=[x, y, z],
                        position_error=error,
                        angle_error=angle,
                    )
                )
    (args.output / "reachable.json").write_text(json.dumps(reachable, indent=2) + "\n")
    print("reachable", len(reachable), flush=True)
    cameras += reachable
    rows = []
    gallery = []
    try:
        for layout in ("open", "source_south_goal_west", "source_west_goal_south"):
            for seed in range(args.seeds):
                env.reset(10000 + seed)
                for i, mid in enumerate(cover_ids):
                    env.data.mocap_pos[mid] = (
                        -0.025,
                        (-0.045, 0.055)[i],
                        -1 if layout == "open" else 0,
                    )
                    yaw = 0 if ((layout == "source_south_goal_west") == (i == 0)) else -np.pi / 2
                    env.data.mocap_quat[mid] = scenes.yaw_quat(yaw)
                mujoco.mj_forward(env.model, env.data)
                wrist, w = audit.render("manipulator/wrist_cam")
                for c in cameras:
                    fixed.pos = c["position"]
                    fixed.quat = scenes.look_at_quat(c["position"], c["lookat"])
                    mujoco.mj_forward(env.model, env.data)
                    rgb, m = audit.render("fixed")
                    rows.append(
                        dict(
                            layout=layout, seed=10000 + seed, camera=c["name"], wrist=w, external=m
                        )
                    )
                    if seed == 0 and c["name"] in (
                        "fixed_00",
                        "fixed_01",
                        *[x["name"] for x in reachable],
                    ):
                        gallery.append(
                            mosaic([wrist, rgb], [layout + " wrist", c["name"]], scale=2)
                        )
                if seed % 8 == 0:
                    print(layout, seed, flush=True)
        summaries = []
        for layout in ("open", "source_south_goal_west", "source_west_goal_south", "mixed"):
            for c in cameras:
                group = [
                    r
                    for r in rows
                    if r["camera"] == c["name"]
                    and (r["layout"] == layout or layout == "mixed" and r["layout"] != "open")
                ]
                if not group:
                    continue

                def pair(r, combined):
                    return all(
                        max(r["external"][k], r["wrist"][k] if combined else 0) >= 3
                        for k in ("object_pixels", "fixture_pixels")
                    )

                summaries.append(
                    dict(
                        layout=layout,
                        camera=c["name"],
                        samples=len(group),
                        external_pair=float(np.mean([pair(r, False) for r in group])),
                        combined_pair=float(np.mean([pair(r, True) for r in group])),
                        source_zero=float(
                            np.mean([r["external"]["object_pixels"] == 0 for r in group])
                        ),
                        goal_zero=float(
                            np.mean([r["external"]["fixture_pixels"] == 0 for r in group])
                        ),
                    )
                )
        report = dict(
            experiment=cfg.to_dict(),
            cameras=cameras,
            rows=rows,
            summaries=summaries,
            scope="Optical enclosure prototype; fixed camera pose counterfactuals, not learned success or executed active trajectories.",
        )
        (args.output / "views.json").write_text(json.dumps(report, indent=2) + "\n")
        sheet = Image.new("RGB", (gallery[0].width * 3, sum(x.height for x in gallery[::3])))
        for i, tile in enumerate(gallery):
            sheet.paste(tile, ((i % 3) * tile.width, (i // 3) * tile.height))
        sheet.save(args.output / "views.png")
        for layout in ("open", "mixed"):
            print(
                layout,
                sorted(
                    [s for s in summaries if s["layout"] == layout],
                    key=lambda x: x["combined_pair"],
                    reverse=True,
                )[:8],
                flush=True,
            )
    finally:
        audit.close()


if __name__ == "__main__":
    main()
