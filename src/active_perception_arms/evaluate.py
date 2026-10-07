"""Checkpoint evaluation and causal interventions; never updates model parameters."""

import argparse
import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from rsl_rl.utils import resolve_callable

from .config import OCCLUSIONS, saved_experiment
from .environment import make_env


def load_actor(checkpoint, observations, device):
    config = json.loads(checkpoint.with_name("runner.json").read_text())
    options = copy.deepcopy(config["actor"])
    cls = resolve_callable(options.pop("class_name"))
    options = {k: v for k, v in options.items() if v is not None}
    experiment = saved_experiment(json.loads(checkpoint.with_name("experiment.json").read_text()))
    actor = cls(observations, config["obs_groups"], "actor", experiment.action_dim, **options).to(
        device
    )
    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(weights["actor_state_dict"])
    return actor.eval()


def wilson(successes, total):
    if total < 1:
        raise ValueError("At least one evaluation episode is required")
    z = 1.95996398454
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def evaluate(args):
    if args.episodes < 1 or args.num_envs < 1:
        raise ValueError("episodes and num-envs must be positive")
    original = saved_experiment(
        json.loads(args.checkpoint.with_name("experiment.json").read_text())
    )
    cfg = replace(
        original,
        num_envs=min(args.num_envs, args.episodes),
        seed=args.seed,
        occlusion=args.occlusion or original.occlusion,
    )
    if (
        args.freeze_camera_after is not None or args.camera_trace_in or args.camera_trace_out
    ) and not cfg.camera_control:
        raise ValueError("Camera interventions require an active or initial camera checkpoint")
    steps = math.ceil(cfg.episode_seconds / cfg.step_dt)
    batches = math.ceil(args.episodes / cfg.num_envs)
    replay = None
    if args.camera_trace_in:
        with np.load(args.camera_trace_in) as data:
            replay = torch.as_tensor(data["actions"], device=args.device)
            if str(data["task"]) != cfg.task or float(data["step_dt"]) != cfg.step_dt:
                raise ValueError("Replay task/control rate mismatch")
        if replay.shape != (batches, steps, cfg.num_envs, 5):
            raise ValueError("Replay must use identical episode count, num-envs, and horizon")
    traces = []
    outcomes, times, distances = [], [], []
    env = make_env(cfg, args.device)
    try:
        obs, _ = env.reset(seed=args.seed)
        actor = load_actor(args.checkpoint, obs, args.device)
        with torch.inference_mode():
            for batch in range(batches):
                env.action_manager.get_term("arms").camera_frozen = False
                obs, _ = env.reset(seed=args.seed + batch)
                actor.reset()
                finished = torch.zeros(cfg.num_envs, device=args.device, dtype=torch.bool)
                won = torch.zeros_like(finished)
                elapsed = torch.full((cfg.num_envs,), cfg.episode_seconds, device=args.device)
                motion = torch.zeros(cfg.num_envs, device=args.device)
                frozen = None
                recorded = []
                for step in range(steps):
                    t = step * cfg.step_dt
                    if args.reset_memory:
                        actor.reset()
                    if args.hold_external_after is not None and t >= args.hold_external_after:
                        if "external" not in obs:
                            raise ValueError("External image hold requires an external camera")
                        if frozen is None:
                            frozen = obs["external"].clone()
                        obs["external"] = frozen
                    action = actor(obs).clone()
                    if cfg.camera_control:
                        if replay is not None:
                            action[:, cfg.manip_dim :] = replay[batch, step]
                        if args.freeze_camera_after is not None and t >= args.freeze_camera_after:
                            action[:, cfg.manip_dim :] = 0
                            # Zero gimbal deltas would still let iterative IK move.
                            env.action_manager.get_term("arms").camera_frozen = True
                        if args.camera_trace_out:
                            recorded.append(action[:, cfg.manip_dim :].cpu().numpy())
                    before = env.scene["camera_arm"].data.joint_pos.clone()
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated | truncated
                    first = done & ~finished
                    # Termination manager retains this step's flags through automatic reset.
                    won |= first & env.termination_manager.get_term("success")
                    elapsed[first] = (step + 1) * cfg.step_dt
                    delta = (env.scene["camera_arm"].data.joint_pos - before).abs().sum(-1)
                    # Exclude terminal reset jumps from physical-motion diagnostics.
                    motion += delta * (~finished & ~done)
                    finished |= done
                    actor.reset(done)
                count = min(cfg.num_envs, args.episodes - len(outcomes))
                outcomes.extend(won[:count].cpu().tolist())
                times.extend(elapsed[:count].cpu().tolist())
                distances.extend(motion[:count].cpu().tolist())
                if recorded:
                    traces.append(np.stack(recorded))
    finally:
        env.close()
    if args.camera_trace_out:
        args.camera_trace_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.camera_trace_out,
            actions=np.stack(traces),
            task=cfg.task,
            step_dt=cfg.step_dt,
            source_seed=args.seed,
            source_checkpoint=str(args.checkpoint),
        )
    successes = sum(outcomes)
    report = {
        "checkpoint": str(args.checkpoint),
        "experiment": cfg.to_dict(),
        "split": args.split,
        "seed": args.seed,
        "training_seed": original.seed,
        "episodes": len(outcomes),
        "successes": successes,
        "success_rate": successes / len(outcomes),
        "wilson_95": wilson(successes, len(outcomes)),
        "outcomes": outcomes,
        "episode_seconds": times,
        "camera_joint_travel_rad": distances,
        "interventions": {
            "freeze_camera_after": args.freeze_camera_after,
            "hold_external_after": args.hold_external_after,
            "reset_memory": args.reset_memory,
            "camera_trace_in": str(args.camera_trace_in) if args.camera_trace_in else None,
        },
        "training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    return {
        k: v
        for k, v in report.items()
        if k not in ("outcomes", "episode_seconds", "camera_joint_travel_rad")
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--occlusion", choices=OCCLUSIONS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--freeze-camera-after", type=float)
    p.add_argument("--hold-external-after", type=float)
    p.add_argument("--reset-memory", action="store_true")
    p.add_argument("--camera-trace-in", type=Path)
    p.add_argument("--camera-trace-out", type=Path)
    p.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = p.parse_args(argv)
    args.seed = (
        args.seed if args.seed is not None else (10000 if args.split == "validation" else 20000)
    )
    torch.set_num_threads(4 if args.device.startswith("cuda") else 1)
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
