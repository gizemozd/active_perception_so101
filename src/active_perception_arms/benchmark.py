"""Measure simulation/rendering throughput; optional actor inference, no learning."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.utils import resolve_callable

from .config import CONDITIONS, TASKS, Experiment
from .environment import make_env
from .policy import runner_cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, default="plug")
    p.add_argument("--condition", choices=CONDITIONS, default="active")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--actor", action="store_true")
    p.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = p.parse_args(argv)
    if args.steps < 1 or args.warmup < 0 or (args.actor and args.no_render):
        p.error("Use positive steps, nonnegative warmup, and rendered images for actor inference")
    cfg = Experiment(
        task=args.task,
        condition=args.condition,
        num_envs=args.num_envs,
        render_sensors=not args.no_render,
    )
    cuda = args.device.startswith("cuda")
    torch.set_num_threads(4 if cuda else 1)
    torch.set_float32_matmul_precision("high")
    env = make_env(cfg, args.device)
    actor = None
    try:
        obs, _ = env.reset()
        if args.actor:
            rc = asdict(runner_cfg(cfg))
            options = rc["actor"]
            cls = resolve_callable(options.pop("class_name"))
            actor = (
                cls(obs, rc["obs_groups"], "actor", cfg.action_dim, **options)
                .to(args.device)
                .eval()
            )
        zero = torch.zeros(cfg.num_envs, cfg.action_dim, device=args.device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                obs, _, term, trunc, _ = env.step(actor(obs) if actor else zero)
                if actor:
                    actor.reset(term | trunc)
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.steps):
                obs, reward, term, trunc, _ = env.step(actor(obs) if actor else zero)
                if actor:
                    actor.reset(term | trunc)
            if cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        if not torch.isfinite(reward).all():
            raise RuntimeError("Non-finite rewards during benchmark")
        report = {
            "experiment": cfg.to_dict(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device) if cuda else None,
            "warmup_steps": args.warmup,
            "timed_steps": args.steps,
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": cfg.num_envs * args.steps / elapsed,
            "actor_inference": args.actor,
            "training_started": False,
            "torch_peak_bytes": torch.cuda.max_memory_allocated() if cuda else None,
            "note": "Torch memory excludes Warp allocations; inspect nvidia-smi for total VRAM. Inference only; PPO update cost excluded.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
