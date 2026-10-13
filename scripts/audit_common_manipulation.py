"""Privileged physical feasibility of matched absolute/delta plug controls."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from active_perception_arms import plug
from active_perception_arms.config import Experiment
from active_perception_arms.native import NativeEnv


def main():
    torch.set_num_threads(1)
    rows = []
    for mode in ("absolute", "delta"):
        for variant in plug.VARIANTS:
            cfg = Experiment(
                task="plug",
                condition="wrist_static",
                control_revision="pose_v1",
                translation_control=mode,
                plug_variant=variant,
                render_sensors=False,
            )
            env = NativeEnv(cfg)
            for seed in range(140000, 140008):
                env.reset(seed)
                initial = env.tcp_target.copy()
                goal = env.goal + plug.GRIP_OFFSET
                stage = 0
                held = 0
                success = False
                for _ in range(88):
                    desired = goal.copy()
                    if stage == 0:
                        desired[:2] = initial[:2]
                        desired[2] = 0.09
                        if env.data.time >= 1.2 and env.tcp[2] >= 0.087:
                            stage = 1
                    elif stage == 1:
                        desired[2] = 0.09
                        if np.linalg.norm(env.tcp[:2] - goal[:2]) < 0.002:
                            stage = 2
                    else:
                        desired[2] = max(goal[2], env.tcp_target[2] - 0.0015)
                    if mode == "delta":
                        action = (desired - env.tcp_target) / cfg.translation_step
                    else:
                        action = (
                            2
                            * (desired - plug.ACTION_LOW)
                            / (np.asarray(plug.ACTION_HIGH) - plug.ACTION_LOW)
                            - 1
                        )
                    won = env.step(action)
                    held = held + 1 if won else 0
                    if held >= 3:
                        success = True
                        break
                rows.append(
                    dict(
                        mode=mode,
                        variant=variant,
                        seed=seed,
                        success=success,
                        stage=stage,
                        seconds=env.data.time,
                        error_mm=float(np.linalg.norm(env.object_pos - env.goal) * 1000),
                    )
                )
        print(mode, sum(r["success"] for r in rows if r["mode"] == mode), "/32", flush=True)
    output = Path("artifacts/common_control_study/manipulation_feasibility.json")
    result = dict(
        scope="Native MuJoCo, privileged lift-align-descend controller; physical feasibility only, no learned-policy claim.",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
