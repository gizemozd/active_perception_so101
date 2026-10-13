#!/usr/bin/env python3
"""Exercise rendered GPU PPO, optional yaw, and checkpoint loading before training."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.rl import RslRlVecEnvWrapper

from active_perception_arms import plug
from active_perception_arms.config import Experiment
from active_perception_arms.environment import make_env
from active_perception_arms.evaluate import load_actor
from active_perception_arms.policy import CompactPPO, runner_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "An allocated CUDA GPU is required"
    torch.set_num_threads(4)
    results = []
    for yaw in (False, True):
        cfg = Experiment(
            condition="active" if yaw else "wrist_static",
            num_envs=8,
            policy_profile="legacy_ff",
            memory="none",
            plug_wrist_yaw=yaw,
            reward_profile="legacy_log_hold",
            randomize=False,
        )
        env = make_env(cfg, "cuda:0")
        try:
            wrapped = RslRlVecEnvWrapper(env)
            obs, _ = wrapped.reset()
            options = asdict(runner_cfg(cfg, iterations=2))
            options["multi_gpu"] = None
            for name in ("actor", "critic"):
                options[name] = {k: v for k, v in options[name].items() if v is not None}
            alg = CompactPPO.construct_algorithm(obs, wrapped, options, "cuda:0")
            before = {k: v.clone() for k, v in alg.actor.cnns.state_dict().items()}
            metrics = []
            for _ in range(2):
                with torch.inference_mode():
                    for _ in range(24):
                        action = alg.act(obs)
                        obs, rewards, dones, info = wrapped.step(action)
                        assert torch.isfinite(rewards).all() and torch.isfinite(action).all()
                        for camera in cfg.sensors:
                            rgb = obs[camera]
                            assert rgb.shape == (8, 3, 96, 128) and rgb.dtype == torch.uint8
                            assert (
                                rgb.flatten(1).max(1).values > rgb.flatten(1).min(1).values
                            ).all()
                        alg.process_env_step(obs, rewards, dones, info)
                    alg.compute_returns(obs)
                metric = alg.update()
                assert all(math.isfinite(v) for v in metric.values())
                metrics.append(metric)
            for camera in cfg.sensors:
                assert any(
                    not torch.equal(v, before[k])
                    for k, v in alg.actor.cnns.state_dict().items()
                    if k.startswith(camera)
                )
            directory = args.output.parent / f"smoke-yaw{int(yaw)}"
            directory.mkdir(exist_ok=True, parents=True)
            (directory / "experiment.json").write_text(json.dumps(cfg.to_dict(), indent=2))
            (directory / "runner.json").write_text(json.dumps(options, indent=2))
            checkpoint = directory / "model.pt"
            torch.save({"actor_state_dict": alg.actor.state_dict()}, checkpoint)
            restored = load_actor(checkpoint, obs, "cuda:0")
            with torch.inference_mode():
                torch.testing.assert_close(restored(obs), alg.actor(obs))
            if yaw:
                # Exercise compiled yaw IK and ensure camera channels do not shift.
                env.reset()
                env.episode_length_buf[:] = 25
                term = env.action_manager.get_term("arms")
                action = torch.zeros(8, cfg.action_dim, device="cuda")
                action[:, :3] = (
                    2
                    * (term.tcp_target - term.plug_ik.low)
                    / (term.plug_ik.high - term.plug_ik.low)
                    - 1
                )
                action[:, 3] = 1
                gimbal = term.gimbal_target.clone()
                env.step(action)
                torch.testing.assert_close(term.gimbal_target, gimbal)
                torch.testing.assert_close(
                    term.wrist_yaw_target, torch.full((8,), plug.WRIST_YAW_MAX_DELTA, device="cuda")
                )
            results.append(
                {
                    "yaw": yaw,
                    "condition": cfg.condition,
                    "passed": True,
                    "updates": 2,
                    "transitions": 384,
                    "metrics": metrics,
                    "checkpoint_roundtrip": True,
                    "both_rgb_encoders_updated": True,
                }
            )
        finally:
            env.close()
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "cases": results,
                "passed": True,
                "competence_claim": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
