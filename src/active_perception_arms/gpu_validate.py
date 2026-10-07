"""Full-resolution restored-plug actor/observation checks on all sensing conditions."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.utils import resolve_callable

from . import mdp, plug
from .config import Experiment, static_candidates
from .environment import make_env
from .policy import runner_cfg


def main():
    assert torch.cuda.is_available(), "CUDA validation must not silently skip"
    torch.set_num_threads(4)
    results = []
    for condition in ("wrist", "wrist_static", "initial", "active"):
        cfg = Experiment(
            condition=condition,
            num_envs=8,
            fixed_position=static_candidates("plug")[7 if condition == "wrist_static" else 0],
        )
        env = make_env(cfg, "cuda:0")
        try:
            obs, _ = env.reset()
            state = mdp.state(env)
            torch.testing.assert_close(
                state.plug_offsets, torch.tensor(list(plug.OFFSETS.values()) * 2, device=env.device)
            )
            before = mdp.proprio(env).clone()
            saved_offsets = state.plug_offsets.clone()
            state.plug_offsets += 1
            state.onset += 0.5
            torch.testing.assert_close(before, mdp.proprio(env))
            state.plug_offsets.copy_(saved_offsets)
            rc = asdict(runner_cfg(cfg))
            assert rc["obs_groups"]["actor"] == ("proprio", *cfg.sensors)
            assert "critic" not in rc["obs_groups"]["actor"]
            options = rc["actor"]
            cls = resolve_callable(options.pop("class_name"))
            actor = cls(obs, rc["obs_groups"], "actor", cfg.action_dim, **options).cuda().eval()
            with torch.inference_mode():
                for _ in range(4):
                    action = actor(obs)
                    assert torch.isfinite(action).all()
                    obs, reward, term, trunc, _ = env.step(action)
                    actor.reset(term | trunc)
                    assert torch.isfinite(reward).all()
                    for name in cfg.sensors:
                        assert obs[name].shape == (8, 3, 96, 128)
                        assert obs[name].is_cuda and obs[name].dtype == torch.uint8
                        assert (
                            obs[name].flatten(1).max(1).values > obs[name].flatten(1).min(1).values
                        ).all()
            if condition == "initial":
                arms = env.action_manager.get_term("arms")
                env.episode_length_buf[:] = 26
                targets, gimbal = arms.targets["camera_arm"].clone(), arms.gimbal_target.clone()
                arms.process_actions(torch.ones(8, 8, device=env.device))
                torch.testing.assert_close(arms.targets["camera_arm"], targets)
                torch.testing.assert_close(arms.gimbal_target, gimbal)
            results.append(
                {
                    "condition": condition,
                    "passed": True,
                    "variants": plug.VARIANTS,
                    "resolution": [128, 96],
                    "actor_privileged_state": False,
                }
            )
        finally:
            env.close()
    Path("artifacts/cluster_pilot/gpu_pipeline.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
