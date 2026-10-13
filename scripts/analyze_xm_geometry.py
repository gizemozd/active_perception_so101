#!/usr/bin/env python3
"""Reconstruct prong/socket geometry from captured poses; no physics replay."""

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from active_perception_arms import plug
from active_perception_arms.config import saved_experiment
from active_perception_arms.native import NativeEnv
from active_perception_arms.sanity import mosaic


def aperture_clearance(vertices, hole_xy, half_width=0.006):
    """Clip box edges to socket's 0..14 mm slab, then measure aperture margin."""
    points = [v for v in vertices if 0 <= v[2] <= 0.014]
    signs = list(itertools.product((-1, 1), repeat=3))
    for i in range(8):
        for j in range(i + 1, 8):
            if sum(a != b for a, b in zip(signs[i], signs[j], strict=True)) != 1:
                continue
            a, b = vertices[i], vertices[j]
            if abs(b[2] - a[2]) < 1e-12:
                continue
            for z in (0, 0.014):
                alpha = (z - a[2]) / (b[2] - a[2])
                if 0 <= alpha <= 1:
                    points.append(a + alpha * (b - a))
    if not points:
        return None
    return (half_width - np.abs(np.array(points)[:, :2] - hole_xy).max(0)).tolist()


def analyze(directory, output, name):
    capture = json.loads((directory / "capture.json").read_text())
    cfg = saved_experiment(capture["experiment"])
    records = []
    for path in sorted(directory.glob("*.npz")):
        data = np.load(path)
        variant = path.name.split("_")[0]
        env = NativeEnv(replace(cfg, plug_variant=variant, num_envs=1, render_sensors=False))
        adr = env.object_adr
        rows = []
        for i in range(len(data["qpos"])):
            env.data.qpos[:] = data["qpos"][i]
            env.data.qvel[:] = data["qvel"][i]
            origin = data["origins"][i]
            env.data.qpos[adr : adr + 3] -= origin
            for body, key in [("fixture/fixture", "fixture"), ("occluder/panel", "panel")]:
                env.data.mocap_pos[env.model.body(body).mocapid[0]] = data[key][i] - origin
            mujoco.mj_forward(env.model, env.data)
            body = env.data.qpos[adr : adr + 3].copy()
            goal = env.goal
            matrix = np.zeros(9)
            mujoco.mju_quat2Mat(matrix, env.data.qpos[adr + 3 : adr + 7])
            matrix = matrix.reshape(3, 3)
            pitch = np.arcsin(np.clip(-matrix[2, 0], -1, 1))
            roll = np.arctan2(matrix[2, 1], matrix[2, 2])
            yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
            contacts = []
            for contact in env.data.contact[: env.data.ncon]:
                names = [env.model.geom(int(g)).name for g in contact.geom]
                if any(n.startswith("object/") for n in names) and any(
                    n.startswith("fixture/") for n in names
                ):
                    contacts.append(dict(geoms=names, distance=float(contact.dist)))
            socket = (data["fixture"][i] - origin)[:2]
            margins = []
            dx, dy = plug.OFFSETS[variant]
            for sign in (1, -1):
                center = np.array([dx + sign * plug.HALF_SPACING, dy, -plug.PRONG_HALF[2]])
                corners = (
                    np.array(list(itertools.product((-1, 1), repeat=3))) * plug.PRONG_HALF + center
                )
                world = corners @ matrix.T + body
                margins.append(aperture_clearance(world, socket + [sign * plug.HALF_SPACING, 0]))
            rows.append(
                dict(
                    step=i,
                    time_s=i * cfg.step_dt,
                    body_error=(body - goal).tolist(),
                    body_error_m=float(np.linalg.norm(body - goal)),
                    rpy_deg=np.rad2deg([roll, pitch, yaw]).tolist(),
                    prong_aperture_margin_xy_m=margins,
                    reconstructed_contacts=contacts,
                    action=data["action"][i].tolist(),
                )
            )
        near = min(rows[25:], key=lambda r: r["body_error_m"])
        record = dict(
            policy=name,
            variant=variant,
            trace=str(path),
            outcome=path.stem.split("_")[1],
            closest=near,
            final=rows[-1],
            rows=rows,
        )
        records.append(record)
        if variant == "xm":
            selected = [24, 40, near["step"], len(rows) - 1]
            tiles = []
            for i in selected:
                tiles.append(
                    mosaic(
                        [
                            data["wrist"][i].transpose(1, 2, 0),
                            data["external"][i].transpose(1, 2, 0),
                        ],
                        [
                            f"{name} wrist t={i * cfg.step_dt:.2f}",
                            f"fixed; error={1000 * rows[i]['body_error_m']:.1f}mm",
                        ],
                        scale=3,
                    )
                )
            gallery = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)))
            for i, tile in enumerate(tiles):
                gallery.paste(tile, (0, i * tile.height))
            gallery.save(output / f"{name}-xm-rgb.png")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {
        "progress_gru": args.root
        / "artifacts/plug_analysis/exploration_repair/progress/evaluation-repeat1-traces",
        **{
            name: args.root / "artifacts/plug_repair" / name / "traces"
            for name in ["legacy_ff_yaw0", "legacy_ff_yaw1"]
        },
    }
    records = []
    for name, path in cases.items():
        records.extend(analyze(path, args.output, name))
    report = dict(
        scope="Representative captured trajectories, not aggregate population estimates. Native contact geometry reconstructed at actual Warp poses; forces and native dynamics are not replayed.",
        records=records,
    )
    (args.output / "geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    for r in records:
        if r["variant"] == "xm":
            print(json.dumps({k: v for k, v in r.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
