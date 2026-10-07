"""Measure simulation/rendering throughput; optional actor inference, no learning."""

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
from rsl_rl.utils import resolve_callable

from .config import CONDITIONS, TASKS, Experiment, static_candidates
from .environment import make_env
from .policy import runner_cfg
from .telemetry import GpuSampler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, default="plug")
    p.add_argument("--condition", choices=CONDITIONS, default="active")
    p.add_argument("--fixed-view", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
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
        fixed_position=static_candidates(args.task)[args.fixed_view],
    )
    cuda = args.device.startswith("cuda")
    torch.set_num_threads(4 if cuda else 1)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "active-perception-so101"),
            group="plug-inference-benchmark-20261007",
            job_type="benchmark-inference",
            tags=["disposable", "inference-only", cfg.condition],
            name=f"{cfg.task}_{cfg.condition}_v{args.fixed_view}_{cfg.memory}_s0_n{cfg.num_envs}",
            config={
                "experiment": cfg.to_dict(),
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            },
        )
    sampler = GpuSampler(args.output.with_suffix(".gpu.csv")).start() if cuda else None
    init_start = time.perf_counter()
    env = make_env(cfg, args.device)
    if cuda:
        torch.cuda.synchronize()
    init_seconds = time.perf_counter() - init_start
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
        warmup_start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.warmup):
                obs, _, term, trunc, _ = env.step(actor(obs) if actor else zero)
                if actor:
                    actor.reset(term | trunc)
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            warmup_seconds = time.perf_counter() - warmup_start
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
            "kind": "simulation_and_actor_inference_only",
            "wandb_url": run.url if run else None,
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch_cuda": torch.version.cuda,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "packages": {
                name: version(name)
                for name in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")
            },
            "environment_initialization_seconds": init_seconds,
            "warmup_seconds": warmup_seconds,
            "actual_transitions": cfg.num_envs * args.steps,
            "units": "One vectorized control step advances N environments = N environment transitions; 20 physics substeps/control step.",
            "gpu_metrics": sampler.metrics() if sampler else {},
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
        if run:
            run.config.update(report)
            run.log(
                {
                    "benchmark/transitions": cfg.num_envs * args.steps,
                    "benchmark/inference_transitions_per_second": report[
                        "environment_steps_per_second"
                    ],
                    "benchmark/measured_seconds": elapsed,
                    **(sampler.metrics() if sampler else {}),
                }
            )
            run.finish()
    finally:
        if sampler:
            sampler.close()
        env.close()


if __name__ == "__main__":
    main()
