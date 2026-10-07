"""Explicit training CLI. --dry-run validates configuration without running training."""

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .config import CONDITIONS, OCCLUSIONS, TASKS, Experiment, saved_experiment, static_candidates


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--condition", choices=CONDITIONS, default="active")
    p.add_argument(
        "--occlusion", choices=OCCLUSIONS, help="Default: clean for plug, random otherwise"
    )
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--initial-seconds", type=float, default=1.0)
    p.add_argument("--episode-seconds", type=float)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument(
        "--fixed-view", type=int, default=0, help="Index in the published static camera search grid"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-root", type=Path, default=Path("logs"))
    p.add_argument("--resume", type=Path)
    p.add_argument("--perturb-push", action="store_true")
    p.add_argument("--memory", choices=("gru", "none"), default="gru")
    p.add_argument("--job-type", choices=("benchmark", "pilot", "study"), default="pilot")
    p.add_argument("--warmup-iterations", type=int, default=3)
    p.add_argument("--result", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p


def experiment_from_args(args):
    candidates = static_candidates(args.task)
    if not 0 <= args.fixed_view < len(candidates):
        raise ValueError(f"fixed-view must be in [0, {len(candidates) - 1}]")
    if args.warmup_iterations < 0:
        raise ValueError("warmup-iterations must be nonnegative")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.num_envs < 8 or args.num_envs % 8:
        raise ValueError("Training num-envs must be a positive multiple of 8 minibatches")
    if args.fixed_view and args.condition not in ("static", "wrist_static"):
        raise ValueError("fixed-view only applies to static camera conditions")
    return Experiment(
        task=args.task,
        condition=args.condition,
        occlusion=args.occlusion,
        num_envs=args.num_envs,
        seed=args.seed,
        width=args.width,
        height=args.height,
        initial_seconds=args.initial_seconds,
        episode_seconds=args.episode_seconds,
        fixed_position=candidates[args.fixed_view],
        perturb_push=args.perturb_push,
        memory=args.memory,
    )


def main(argv=None):
    process_start = time.perf_counter()
    args = parser().parse_args(argv)
    cfg = experiment_from_args(args)
    if args.resume:
        original = saved_experiment(
            json.loads(args.resume.with_name("experiment.json").read_text())
        ).to_dict()
        # A resume restores the same experiment; interventions belong in evaluation.
        for key, value in cfg.to_dict().items():
            if key != "num_envs" and json.dumps(value) != json.dumps(original[key]):
                raise ValueError(f"Resume configuration differs in {key}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment": cfg.to_dict(),
                    "iterations": args.iterations,
                    "training_started": False,
                },
                indent=2,
            )
        )
        return
    import torch
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from .environment import make_env
    from .policy import runner_cfg

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA GPU. Use --dry-run or arms-sanity locally.")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{cfg.task}_{cfg.condition}_{cfg.occlusion}_{cfg.memory}_v{args.fixed_view}_s{cfg.seed}_n{cfg.num_envs}_{stamp}"
    log_dir = args.resume.parent if args.resume else args.log_root / name
    os.environ.setdefault("WANDB_PROJECT", "active-perception-so101")
    os.environ.setdefault("WANDB_RUN_GROUP", f"plug-{args.job_type}-20261007")
    os.environ.setdefault("WANDB_JOB_TYPE", args.job_type)
    os.environ.setdefault(
        "WANDB_TAGS", f"restored-plug,{args.job_type},{cfg.condition},{cfg.memory}"
    )
    if args.resume:
        run_info = json.loads((log_dir / "wandb_run.json").read_text())
        os.environ["WANDB_RUN_ID"] = run_info["id"]
        os.environ["WANDB_RESUME"] = "must"
        os.environ["WANDB_PROJECT"] = run_info["project"]
        os.environ["WANDB_ENTITY"] = run_info["entity"]
    log_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    (log_dir / "experiment.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    runner_options = runner_cfg(cfg, args.iterations)
    runner_options.logger = "wandb"
    runner_options.wandb_project = os.environ["WANDB_PROJECT"]
    runner_options.save_interval = max(1, args.iterations // 2)
    (log_dir / "runner.json").write_text(json.dumps(asdict(runner_options), indent=2))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    metadata = {
        "command": sys.argv,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(args.device),
        "torch_cuda": torch.version.cuda,
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "packages": {
            name: version(name)
            for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib")
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "resume": str(args.resume) if args.resume else None,
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "hostname": platform.node(),
        "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "allocated_memory_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (log_dir / "runtime.json").write_text(json.dumps(metadata, indent=2))
    from .telemetry import GpuSampler, install_logging

    sampler = GpuSampler(log_dir / "gpu.csv").start()
    init_start = time.perf_counter()
    env = make_env(cfg, args.device)
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - init_start
    try:
        wrapped = RslRlVecEnvWrapper(env)
        runner = MjlabOnPolicyRunner(wrapped, asdict(runner_options), str(log_dir), args.device)
        if args.resume:
            runner.load(str(args.resume))
            runner.current_learning_iteration += 1
        remaining = args.iterations - (runner.current_learning_iteration if args.resume else 0)
        install_logging(runner, cfg, args, log_dir, metadata, init_seconds, sampler, process_start)
        interrupted = []

        def request_checkpoint(signum, frame):
            interrupted.append(signum)

        # Save after a complete PPO iteration, outside the signal handler. A Slurm
        # USR1 signal gives the job time to flush a resumable checkpoint.
        original_log = runner.logger.log

        def log_and_checkpoint(**kwargs):
            original_log(**kwargs)
            if interrupted:
                runner.save(str(log_dir / "interrupted.pt"))
                runner.logger.stop_logging_writer()
                raise SystemExit(0)

        runner.logger.log = log_and_checkpoint
        signal.signal(signal.SIGUSR1, request_checkpoint)
        signal.signal(signal.SIGTERM, request_checkpoint)
        # This is the sole learning call. It is never used by tests or sanity tools.
        if remaining > 0:
            runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
    finally:
        sampler.close()
        env.close()


if __name__ == "__main__":
    main()
