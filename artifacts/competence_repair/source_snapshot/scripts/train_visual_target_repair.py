"""Learn an RGB target estimator with supervised simulator labels.

Deployment uses the saved estimator and a prescribed lift-align-descend motion.
Autonomous evaluation never receives target labels or a teacher controller.
"""

import argparse
import copy
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.utils import resolve_callable

from active_perception_arms import mdp, plug
from active_perception_arms.config import Experiment, static_candidates
from active_perception_arms.environment import make_env
from active_perception_arms.policy import runner_cfg


def save(path, actor, cfg, options, step):
    path.mkdir(exist_ok=True, parents=True)
    (path / "experiment.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (path / "runner.json").write_text(json.dumps(options, indent=2) + "\n")
    checkpoint = path / f"model_{step}.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "iter": step,
            "method": "supervised RGB target estimator with prescribed insertion motion",
        },
        checkpoint,
    )
    return checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", default="all", choices=["all", "xm", "xp", "ym", "yp"])
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--horizon", type=float, default=8)
    p.add_argument("--clearance", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jitter", type=float, default=0.008)
    p.add_argument("--occlusion", choices=["clean", "dynamic"], default="clean")
    p.add_argument("--eval-dynamic", action="store_true")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--dense", action="store_true")
    p.add_argument("--inspection-only", action="store_true")
    p.add_argument("--visual-data-only", action="store_true")
    p.add_argument("--factorized", action="store_true")
    p.add_argument("--initialize-actor", type=Path)
    p.add_argument("--eval-interval", type=int, default=2000)
    args = p.parse_args()
    if (args.factorized or args.visual_data_only) and not args.inspection_only:
        p.error("--factorized and --visual-data-only require --inspection-only")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cfg = Experiment(
        task="plug",
        condition="wrist_static",
        occlusion=args.occlusion,
        num_envs=args.num_envs,
        seed=args.seed,
        episode_seconds=args.horizon,
        clearance=args.clearance,
        reset_xy_jitter=args.jitter,
        fixed_position=static_candidates("plug")[7],
        plug_variant=None if args.variant == "all" else args.variant,
    )
    (args.output / "method.json").write_text(
        json.dumps(
            dict(
                method="supervised visual target estimation with prescribed lift-align-descend",
                teacher="privileged target labels during training only; actor uses RGB/proprio and a time-based motion plan",
                actor_inputs=["wrist RGB", "external RGB", "proprioception"],
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            ),
            indent=2,
        )
    )
    env = make_env(cfg, "cuda:0")
    try:
        obs, _ = env.reset()

        options = asdict(runner_cfg(cfg))
        options["actor"]["class_name"] = (
            "active_perception_arms.visual_target_policy:VisualTargetPolicy"
        )
        if args.dense:
            options["actor"]["class_name"] = (
                "active_perception_arms.visual_target_policy:DenseVisualTargetPolicy"
            )
        if args.inspection_only:
            options["actor"]["obs_normalization"] = False
            options["actor"]["class_name"] = (
                "active_perception_arms.visual_target_policy:InspectionVisualTargetPolicy"
            )
        if args.factorized:
            options["actor"]["class_name"] = (
                "active_perception_arms.visual_target_policy:FactorizedInspectionPolicy"
            )
        options["actor"]["episode_seconds"] = args.horizon
        actor_options = copy.deepcopy(options["actor"])
        cls = resolve_callable(actor_options.pop("class_name"))
        actor = cls(obs, options["obs_groups"], "actor", cfg.action_dim, **actor_options).cuda()
        if args.initialize_actor:
            weights = torch.load(args.initialize_actor, map_location="cuda:0", weights_only=True)
            actor.load_state_dict(weights["actor_state_dict"])
        optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
        losses = []
        successes = episodes = 0
        start = time.time()
        for step in range(1, args.steps + 1):
            with torch.no_grad():
                _, _, _, goal = mdp.positions(env)
                term = env.action_manager.get_term("arms")
                target = goal - env.scene.env_origins + goal.new_tensor(plug.GRIP_OFFSET)
                labels = (
                    2 * (target - term.plug_ik.low) / (term.plug_ik.high - term.plug_ik.low) - 1
                )
                actor.update_normalization(obs)
            if args.factorized:
                components = actor.predict_components({k: v.clone() for k, v in obs.items()})
                prediction = actor.target_from_components(components)
                fixture = goal - env.scene.env_origins
                fixture = fixture[:, :2] + mdp.state(env).plug_offsets
                xy_target = (fixture - actor.fixture_center) / 0.008
                classes = torch.tensor(
                    plug.variant_assignment(cfg.num_envs, cfg.plug_variant), device=env.device
                )
                per_example = (components[:, :2] - xy_target).square().mean(
                    -1
                ) + torch.nn.functional.cross_entropy(components[:, 2:], classes, reduction="none")
                valid = (obs["proprio"][:, -1] * args.horizon < 1.5).float()
                loss = (per_example * valid).sum() / valid.sum().clamp_min(1)
            else:
                prediction = actor.predict_target({k: v.clone() for k, v in obs.items()})
            if args.inspection_only and not args.factorized:
                valid = (obs["proprio"][:, -1] * args.horizon < 1.5).float()
                loss = (
                    (prediction - labels).square().mean(-1) * valid
                ).sum() / valid.sum().clamp_min(1)
            elif not args.factorized:
                loss = (prediction - labels).square().mean()
            losses.append(loss)
            # Teacher probability decays on rollout states, never supplies actor observations.
            beta = max(0.15, 1 - (step / 8000))
            with torch.no_grad():
                choose = torch.rand(env.num_envs, 1, device=env.device) < beta
                selected = torch.where(choose, labels, prediction.detach()).clamp(-1, 1)
                if args.visual_data_only:
                    selected = torch.zeros_like(selected)
                action = actor.action_from_target(selected, obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                successes += int(env.termination_manager.get_term("success").sum())
                episodes += int(done.sum())
                # Out-of-place recurrent reset avoids invalidating the BPTT graph.
                if actor.rnn.hidden_state is not None:
                    with torch.enable_grad():
                        actor.rnn.hidden_state = actor.rnn.hidden_state * (~done)[None, :, None]
            if args.visual_data_only and step % 40 == 0:
                with torch.no_grad():
                    obs, _ = env.reset()
                    actor.reset()
            if step % 8 == 0:
                optimizer.zero_grad()
                objective = torch.stack(losses).mean()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
                actor.detach_hidden_state()
                losses = []
            if step % 100 == 0:
                row = dict(
                    step=step,
                    transitions=step * args.num_envs,
                    loss=float(loss.detach()),
                    teacher_probability=beta,
                    training_mixed_successes=successes,
                    training_mixed_episodes=episodes,
                    seconds=time.time() - start,
                )
                with (args.output / "iterations.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                successes = episodes = 0
            if step % args.eval_interval == 0:
                checkpoint = save(args.output, actor, cfg, options, step)
                command = [
                    os.sys.executable,
                    "-m",
                    "active_perception_arms.evaluate",
                    str(checkpoint),
                    "--episodes",
                    "256",
                    "--num-envs",
                    "128",
                    "--seed",
                    "31000",
                    "--split",
                    "validation",
                    "--occlusion",
                    "clean",
                    "--audit-termination",
                    "--output",
                    str(args.output / f"evaluation-{step}.json"),
                ]
                result = subprocess.run(
                    command,
                    stdout=(args.output / f"evaluate-{step}.log").open("w"),
                    stderr=subprocess.STDOUT,
                )
                if result.returncode:
                    raise RuntimeError(f"Evaluation failed: {result.returncode}")
                print(
                    "EVALUATION",
                    (args.output / f"evaluation-{step}.json").read_text()[:300],
                    flush=True,
                )
                if args.eval_dynamic:
                    dynamic_command = list(command)
                    dynamic_command[dynamic_command.index("--occlusion") + 1] = "dynamic"
                    dynamic_command[dynamic_command.index("--output") + 1] = str(
                        args.output / f"evaluation-dynamic-{step}.json"
                    )
                    with (args.output / f"evaluate-dynamic-{step}.log").open("w") as stream:
                        subprocess.run(
                            dynamic_command, stdout=stream, stderr=subprocess.STDOUT, check=True
                        )
        save(args.output, actor, cfg, options, args.steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
