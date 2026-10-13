"""CPU learnability diagnostic: RSL-RL PPO on threaded native plug worlds.

Separates control/reward learnability from perception. The actor receives full
state (``privileged``), the socket but not the hidden prong offset
(``socket_only``) or neither (``blind``). Same RSL-RL PPO implementation and
default hyperparameters as the cluster runs; MLP actor/critic.

Example:
    .venv/bin/python scripts/train_native_privileged.py --revision v2 \
        --iterations 400 --output artifacts/diagnosis/privileged_v2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from active_perception_arms import plug
from active_perception_arms.config import Experiment
from active_perception_arms.native_vec import OBSERVATIONS, NativePlugBatch


def experiment(args, num_envs, seed):
    revision = {"v1": "hidden_prongs_v1", "v2": "hidden_prongs_v2"}[args.revision]
    fields = dict(
        task="plug",
        condition="wrist_static",
        control_revision="pose_v1",
        translation_control=args.translation_control,
        translation_step=args.translation_step,
        task_revision=revision,
        num_envs=num_envs,
        seed=seed,
        render_sensors=False,
    )
    for name in (
        "reward_profile",
        "plug_success_rule",
        "plug_jam_failure",
        "episode_seconds",
        "plug_variant",
    ):
        value = getattr(args, name)
        if value is not None:
            fields[name] = value
    if args.plug_reset_scale is not None:
        fields["plug_reset_scale"] = args.plug_reset_scale
    if getattr(args, "plug_assist_fraction", None) and not getattr(args, "evaluating", False):
        fields["plug_assist_fraction"] = args.plug_assist_fraction
    return Experiment(**fields)


def runner_cfg(args):
    hidden = tuple(args.hidden)
    return {
        "num_steps_per_env": args.steps_per_env,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "multi_gpu": None,
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": hidden,
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": args.init_std,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": hidden,
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
            "num_learning_epochs": args.epochs,
            "num_mini_batches": args.minibatches,
            "learning_rate": args.learning_rate,
            "schedule": "adaptive",
            "entropy_coef": args.entropy_coef,
            "gamma": args.gamma,
            "lam": 0.95,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


class EpisodeStats:
    def __init__(self):
        self.rows = []

    def add(self, info):
        done = info["done"]
        for i in np.flatnonzero(done):
            self.rows.append({k: info[k][i] for k in info if k != "done"})

    def summary(self):
        if not self.rows:
            return {"episodes": 0}
        s = np.array([r["success"] for r in self.rows], dtype=bool)
        variants = np.array([r["variant"] for r in self.rows])
        out = {
            "episodes": len(self.rows),
            "success": float(s.mean()),
            "jam": float(np.mean([r["jam"] for r in self.rows])),
            "timeout": float(np.mean([r["timeout"] for r in self.rows])),
            "xy_error_mm": float(np.mean([r["xy_error"] for r in self.rows]) * 1000),
            "height_mm": float(np.mean([r["height"] for r in self.rows]) * 1000),
            "error_3d_mm": float(np.mean([r["error_3d"] for r in self.rows]) * 1000),
        }
        for k, name in enumerate(plug.VARIANTS):
            mask = variants == k
            if mask.any():
                out[f"success_{name}"] = float(s[mask].mean())
        return out


def evaluate(actor, args, seed, episodes):
    """Deterministic actor; first episode of each world only (balanced variants).

    Evaluation always uses the original reset distribution (no assisted starts).
    """
    import copy

    eval_args = copy.copy(args)
    eval_args.evaluating = True
    env = NativePlugBatch(
        experiment(eval_args, episodes, seed),
        episodes,
        threads=args.threads,
        seed=seed,
        observation=eval_args.observation,
    )
    obs = env.get_observations()
    finished = np.zeros(episodes, dtype=bool)
    stats = EpisodeStats()
    with torch.inference_mode():
        while not finished.all():
            actions = actor(obs, stochastic_output=False)
            obs, _, dones, extras = env.step(actions)
            info = dict(extras["episode"])
            first = info["done"] & ~finished
            info["done"] = first
            stats.add(info)
            finished |= dones.numpy()
    env.close()
    return stats.summary()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", choices=("v1", "v2"), default="v2")
    p.add_argument("--reward-profile")
    p.add_argument("--plug-success-rule", choices=("hover_v1", "seated_v2"))
    p.add_argument("--plug-jam-failure", type=lambda s: s.lower() in ("1", "true", "yes"))
    p.add_argument("--plug-reset-scale", type=float)
    p.add_argument("--plug-assist-fraction", type=float, help="Training resets only")
    p.add_argument("--plug-variant", choices=plug.VARIANTS)
    p.add_argument("--episode-seconds", type=float)
    p.add_argument("--translation-control", choices=("delta", "absolute"), default="delta")
    p.add_argument("--translation-step", type=float, default=0.005)
    p.add_argument("--observation", choices=OBSERVATIONS, default="privileged")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--steps-per-env", type=int, default=24)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--init-std", type=float, default=0.4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--entropy-coef", type=float, default=0.003)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=256)
    p.add_argument("--eval-seed", type=int, default=50000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = experiment(args, args.num_envs, args.seed)
    env = NativePlugBatch(
        cfg, args.num_envs, threads=args.threads, seed=args.seed, observation=args.observation
    )
    from rsl_rl.algorithms import PPO

    train_cfg = runner_cfg(args)
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args) | {"output": str(args.output)},
                "experiment": cfg.to_dict(),
                "runner": train_cfg,
            },
            indent=2,
            default=str,
        )
    )
    alg = PPO.construct_algorithm(env.get_observations(), env, train_cfg, "cpu")
    obs = env.get_observations()
    history = (args.output / "history.jsonl").open("w")
    evaluations = []
    start = time.time()
    transitions = 0
    for it in range(args.iterations):
        stats = EpisodeStats()
        rewards = []
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(args.steps_per_env):
                actions = alg.act(obs)
                obs, reward, dones, extras = env.step(actions)
                alg.process_env_step(obs, reward, dones, extras)
                stats.add(extras["episode"])
                rewards.append(reward.mean().item())
            alg.compute_returns(obs)
        collect = time.time() - t0
        losses = alg.update()
        transitions += args.num_envs * args.steps_per_env
        row = {
            "iteration": it + 1,
            "transitions": transitions,
            "wall_seconds": time.time() - start,
            "collect_seconds": collect,
            "mean_step_reward": float(np.mean(rewards)),
            "learning_rate": alg.learning_rate,
            "action_std": alg.get_policy().distribution.std_param.detach().tolist(),
            **{f"loss_{k}": v for k, v in losses.items()},
            **{f"train_{k}": v for k, v in stats.summary().items()},
        }
        if (it + 1) % args.eval_every == 0 or it + 1 == args.iterations:
            alg.eval_mode()
            result = evaluate(alg.get_policy(), args, args.eval_seed, args.eval_episodes)
            alg.train_mode()
            result |= {"iteration": it + 1, "transitions": transitions}
            evaluations.append(result)
            row["eval"] = result
            torch.save(alg.save() | {"iter": it + 1}, args.output / f"model_{it + 1}.pt")
        history.write(json.dumps(row) + "\n")
        history.flush()
        if (it + 1) % 10 == 0 or "eval" in row:
            print(
                f"it {it + 1:4d} {transitions / 1e6:6.2f}M "
                f"{transitions / (time.time() - start):6.0f}/s "
                f"r {row['mean_step_reward']:7.3f} "
                f"succ {row.get('train_success', float('nan')):.3f} "
                f"jam {row.get('train_jam', float('nan')):.3f} "
                f"xy {row.get('train_xy_error_mm', float('nan')):6.1f}mm "
                f"std {np.round(row['action_std'], 3).tolist()}"
                + (f" | EVAL {row['eval'].get('success', 0):.3f}" if "eval" in row else ""),
                flush=True,
            )
    env.close()
    (args.output / "result.json").write_text(
        json.dumps(
            {
                "evaluations": evaluations,
                "transitions": transitions,
                "wall_seconds": time.time() - start,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
