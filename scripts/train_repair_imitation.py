"""Demonstration-assisted RGB policy repair; privileged teacher is training-only.

Uses online DAgger with recurrent truncated BPTT. Evaluation always loads the
saved RGB/proprio actor through the standard audited evaluator, with no teacher.
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


class PlugTeacher:
    def __init__(self, env):
        self.env = env
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.start = env.action_manager.get_term("arms").tcp_target.clone()

    def __call__(self):
        env = self.env
        term = env.action_manager.get_term("arms")
        fresh = env.episode_length_buf <= 1
        self.phase[fresh] = 0
        self.start[fresh] = term.tcp_target[fresh]
        _, _, _, goal = mdp.positions(env)
        desired = goal - env.scene.env_origins + goal.new_tensor(plug.GRIP_OFFSET)
        q = env.sim.data.qpos[:, term.plug_qpos_addresses]
        tcp, _ = term.plug_ik.manip.fk(q)
        tcp = term.plug_ik.manip_frame.pos_to_world(tcp)
        raised = (tcp[:, 2] > 0.087) & (env.episode_length_buf * env.step_dt >= 1.24)
        self.phase = torch.where((self.phase == 0) & raised, 1, self.phase)
        aligned = ((tcp[:, :2] - desired[:, :2]).norm(dim=-1) < 0.002) & (tcp[:, 2] > 0.084)
        self.phase = torch.where((self.phase == 1) & aligned, 2, self.phase)
        desired[:, :2] = torch.where((self.phase == 0)[:, None], self.start[:, :2], desired[:, :2])
        desired[:, 2] = torch.where(
            self.phase < 2, 0.09, torch.maximum(desired[:, 2], term.tcp_target[:, 2] - 0.001)
        )
        action = torch.zeros(env.num_envs, term.exp.action_dim, device=env.device)
        action[:, :3] = (
            2 * (desired - term.plug_ik.low) / (term.plug_ik.high - term.plug_ik.low) - 1
        )
        return action.clamp(-1, 1)


def save(path, actor, cfg, options, step):
    path.mkdir(exist_ok=True, parents=True)
    (path / "experiment.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (path / "runner.json").write_text(json.dumps(options, indent=2) + "\n")
    checkpoint = path / f"model_{step}.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "iter": step,
            "method": "RGB DAgger; teacher absent at evaluation",
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
    p.add_argument("--teacher-only", action="store_true")
    p.add_argument("--eval-interval", type=int, default=2000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cfg = Experiment(
        task="plug",
        condition="wrist_static",
        occlusion="clean",
        num_envs=args.num_envs,
        seed=args.seed,
        episode_seconds=args.horizon,
        clearance=args.clearance,
        fixed_position=static_candidates("plug")[7],
        plug_variant=None if args.variant == "all" else args.variant,
    )
    (args.output / "method.json").write_text(
        json.dumps(
            dict(
                method="online DAgger",
                teacher="lift-align-descend with privileged goal in training only",
                actor_inputs=["wrist RGB", "external RGB", "proprioception"],
                args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            ),
            indent=2,
        )
    )
    env = make_env(cfg, "cuda:0")
    try:
        obs, _ = env.reset()
        teacher = PlugTeacher(env)
        options = asdict(runner_cfg(cfg))
        actor_options = copy.deepcopy(options["actor"])
        cls = resolve_callable(actor_options.pop("class_name"))
        actor = cls(obs, options["obs_groups"], "actor", cfg.action_dim, **actor_options).cuda()
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        losses = []
        successes = episodes = 0
        start = time.time()
        for step in range(1, args.steps + 1):
            with torch.no_grad():
                labels = teacher()
                actor.update_normalization(obs)
            prediction = actor({k: v.clone() for k, v in obs.items()})
            loss = (prediction - labels).square().mean()
            losses.append(loss)
            # Teacher probability decays on rollout states, never supplies actor observations.
            beta = max(0.15, 1 - (step / 8000))
            with torch.no_grad():
                choose = torch.rand(env.num_envs, 1, device=env.device) < beta
                action = torch.where(choose, labels, prediction.detach()).clamp(-1, 1)
                if args.teacher_only:
                    action = labels
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                successes += int(env.termination_manager.get_term("success").sum())
                episodes += int(done.sum())
                # Out-of-place recurrent reset avoids invalidating the BPTT graph.
                if actor.rnn.hidden_state is not None:
                    with torch.enable_grad():
                        actor.rnn.hidden_state = actor.rnn.hidden_state * (~done)[None, :, None]
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
        save(args.output, actor, cfg, options, args.steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
