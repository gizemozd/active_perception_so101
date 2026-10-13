"""Preflight the new 50–120 mm push layout in GPU Warp and RGB PPO."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from audit_manipulation_study import audit
from audit_transfer_warp import ppo_smoke

from active_perception_arms.config import Experiment
from active_perception_arms.environment import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cfg = Experiment(
        task="push",
        task_revision="push_v2",
        push_target_layout="arc_v2",
        push_visibility_revision="crossbar_v3",
        condition="active",
        reward_profile="push_dense",
        manip_control="cartesian",
        occlusion="clean",
        reset_xy_jitter=0.02,
        num_envs=args.episodes,
        seed=10000,
    )
    env = make_env(replace(cfg, num_envs=8), "cuda:0")
    try:
        obs, _ = env.reset()
        assert obs["wrist"].dtype == obs["external"].dtype == torch.uint8
        assert torch.isfinite(obs["proprio"]).all()
        assert env.sim.mj_model.geom("visibility/crossbar").contype == 0
        assert env.sim.mj_model.geom("visibility/crossbar").conaffinity == 0
        assert env.scene["fixture"].data.root_link_pos_w.shape[0] == 8
    finally:
        env.close()
    physics = audit(cfg, args.output / "physics.json")
    smoke = ppo_smoke("cuda:0", args.output / "ppo-smoke", replace(cfg, num_envs=8))
    assert smoke["passed"]
    result = {
        "push": {
            "successes": physics["terminal_flags"]["success"],
            "episodes": len(physics["records"]),
            "smoke": smoke,
            "interpretation": "Scripted feasibility screen, not a competence gate",
        },
        "experiment": cfg.to_dict(),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
