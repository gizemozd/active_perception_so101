"""Verify the fixed-FOV optical crossbar in the training simulator and physics."""

import argparse
import json
from pathlib import Path

import torch
from audit_manipulation_study import audit
from audit_transfer_warp import ppo_smoke

from active_perception_arms.config import Experiment
from active_perception_arms.environment import make_env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=128)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    base = Experiment(
        task="push",
        condition="active",
        occlusion="clean",
        task_revision="push_v2",
        reward_profile="push_dense",
        manip_control="cartesian",
        reset_xy_jitter=0.02,
        push_visibility_revision="crossbar_v3",
        num_envs=8,
    )
    env = make_env(base, "cuda:0")
    try:
        obs, _ = env.reset()
        assert "wrist" in obs and "external" in obs
        assert obs["wrist"].dtype == torch.uint8 and obs["external"].dtype == torch.uint8
        assert torch.isfinite(obs["proprio"]).all()
        assert env.scene["visibility"] is not None
        assert env.sim.mj_model.geom("visibility/crossbar").contype == 0
        assert env.sim.mj_model.geom("visibility/crossbar").conaffinity == 0
    finally:
        env.close()
    physics_cfg = Experiment(**{**base.to_dict(), "num_envs": args.num_envs})
    physics = audit(physics_cfg, args.output / "push-physics.json")
    smoke = ppo_smoke(
        "cuda:0",
        args.output / "push-smoke",
        physics_cfg.__class__(**{**physics_cfg.to_dict(), "num_envs": 8}),
    )
    result = {
        "push": {
            "successes": physics["terminal_flags"]["success"],
            "episodes": len(physics["records"]),
            "smoke": smoke,
            "warp_rgb_and_collisionless_screen": True,
        }
    }
    assert result["push"]["successes"] / result["push"]["episodes"] >= 0.95
    assert smoke["passed"]
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
