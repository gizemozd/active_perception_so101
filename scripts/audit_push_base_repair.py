"""Static IK comparison of base placement and Cartesian limits for the same arc120 targets."""

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from active_perception_arms.config import Experiment
from active_perception_arms.native import NativeEnv

root = Path(__file__).resolve().parents[1]
out = root / "artifacts/competence_repair/push_workspace"
out.mkdir(exist_ok=True, parents=True)
torch.set_num_threads(2)
cfg = Experiment(
    task="push",
    task_revision="push_v2",
    push_target_layout="arc_v2",
    manip_control="cartesian",
    condition="wrist_static",
    occlusion="clean",
)
env = NativeEnv(cfg)
points = []
for dx, dy, angle, radius in itertools.product(
    [-0.02, 0, 0.02], [-0.02, 0, 0.02], np.linspace(-80, 80, 9), [0.05, 0.085, 0.12]
):
    source = np.array([-0.025 + dx, -0.045 + dy])
    direction = np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
    goal = source + radius * direction
    for stage, xy, z in [
        ("approach", source - 0.028 * direction, 0.04),
        ("contact_end", goal - 0.017 * direction, 0.04),
        ("withdraw", goal - 0.017 * direction, 0.075),
    ]:
        points.append(
            dict(source=source.tolist(), goal=goal.tolist(), stage=stage, target=[*xy, z])
        )
target = torch.tensor([p["target"] for p in points], dtype=torch.float64)
chain = env.ik.chains["manipulator"]
rotation = env.ik.rotation[None].expand(len(points), -1, -1)
rows = []
for bx in [-0.19, -0.17, -0.15, -0.13]:
    base = target.new_tensor([bx, 0.025, 0.05])
    q = chain.solve_ik_staged(
        target - base, rotation, site_axis=(1, 0, 0), iters_axis=100, iters_full=150, damping=0.006
    )
    xyz, rot = chain.fk(q)
    err = (xyz + base - target).norm(dim=-1)
    row = dict(
        base_x=bx,
        points=len(points),
        within_2mm=int((err < 0.002).sum()),
        max_error_mm=float(err.max() * 1000),
        failures=[
            dict(**p, error_mm=float(e * 1000))
            for p, e in zip(points, err, strict=True)
            if e >= 0.002
        ],
    )
    rows.append(row)
    print({k: v for k, v in row.items() if k != "failures"}, flush=True)
(out / "base_ik.json").write_text(json.dumps(rows, indent=2) + "\n")
