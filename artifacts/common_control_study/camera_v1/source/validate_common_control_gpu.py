"""Rendered CUDA PPO and matched fixed/active starting-view gate."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.rl import RslRlVecEnvWrapper

from active_perception_arms.config import Experiment
from active_perception_arms.environment import make_env
from active_perception_arms.evaluate import load_actor
from active_perception_arms.policy import CompactPPO, runner_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inspection", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for task in ("plug", "push"):
        reference = None
        for condition, bounded in (("wrist_static", False), ("active", False), ("active", True)):
            cfg = Experiment(
                task=task,
                condition=condition,
                control_revision="pose_v1",
                camera_workspace="inspection_v2" if args.inspection else "local_v1",
                camera_start="inspection_v1" if args.inspection else "home",
                num_envs=8,
                width=128,
                height=96,
                occlusion="clean",
            )
            env = make_env(cfg, "cuda:0")
            try:
                wrapped = RslRlVecEnvWrapper(env)
                obs, _ = wrapped.reset()
                images = {k: obs[k].clone() for k in cfg.sensors}
                if reference is None:
                    reference = images
                else:
                    for k in images:
                        torch.testing.assert_close(images[k], reference[k], rtol=0, atol=0)
                options = asdict(runner_cfg(cfg, 2))
                options["multi_gpu"] = None
                if bounded:
                    options["actor"]["distribution_cfg"].update(
                        class_name="active_perception_arms.bounded_distribution:SquashedGaussian",
                        init_std=0.3,
                        std_range=(0.05, 1.0),
                    )
                for name in ("actor", "critic"):
                    options[name] = {k: v for k, v in options[name].items() if v is not None}
                alg = CompactPPO.construct_algorithm(obs, wrapped, options, "cuda:0")
                before = {k: v.clone() for k, v in alg.actor.cnns.state_dict().items()}
                metrics = []
                for _ in range(2):
                    with torch.inference_mode():
                        for _ in range(24):
                            action = alg.act(obs)
                            if bounded:
                                assert action.abs().max() <= 1
                            obs, reward, dones, info = wrapped.step(action)
                            assert torch.isfinite(action).all() and torch.isfinite(reward).all()
                            alg.process_env_step(obs, reward, dones, info)
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
                directory = args.output.parent / f"{task}_{condition}_{bounded}"
                directory.mkdir(exist_ok=True)
                (directory / "experiment.json").write_text(json.dumps(cfg.to_dict()))
                (directory / "runner.json").write_text(json.dumps(options))
                checkpoint = directory / "model.pt"
                torch.save({"actor_state_dict": alg.actor.state_dict()}, checkpoint)
                restored = load_actor(checkpoint, obs, "cuda:0")
                alg.actor.reset()
                with torch.inference_mode():
                    torch.testing.assert_close(restored(obs), alg.actor(obs))
                results.append(
                    dict(
                        task=task,
                        condition=condition,
                        bounded=bounded,
                        metrics=metrics,
                        passed=True,
                        image_hash_equivalence=True,
                        both_encoders_updated=True,
                        checkpoint_roundtrip=True,
                    )
                )
            finally:
                env.close()
    args.output.write_text(
        json.dumps(dict(passed=True, gpu=torch.cuda.get_device_name(), results=results), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
