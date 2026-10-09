"""Explicit training CLI. --dry-run validates configuration without running training."""

import argparse
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .config import (
    CONDITIONS,
    OCCLUSIONS,
    SUCCESS_STATE_SAMPLES,
    TASKS,
    Experiment,
    saved_experiment,
    static_candidates,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--condition", choices=CONDITIONS, default="active")
    p.add_argument(
        "--occlusion", choices=OCCLUSIONS, help="Default: clean for plug, random otherwise"
    )
    p.add_argument("--dynamic-onset-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    p.add_argument("--dynamic-duration-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    p.add_argument("--dynamic-center", type=float, nargs=3, metavar=("X", "Y", "Z"))
    p.add_argument("--dynamic-center-jitter", type=float, nargs=3, metavar=("X", "Y", "Z"))
    p.add_argument("--dynamic-travel-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    p.add_argument("--dynamic-panel-half-size", type=float, nargs=3, metavar=("X", "Y", "Z"))
    p.add_argument("--dynamic-panel-yaw", type=float, help="World-space panel yaw in radians")
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
    p.add_argument(
        "--success-state-sample",
        choices=SUCCESS_STATE_SAMPLES,
        help="Fresh default: current_qpos; resume inherits saved sampling",
    )
    p.add_argument("--perturb-push", action="store_true")
    p.add_argument("--memory", choices=("gru", "none"), default="gru")
    p.add_argument("--job-type", choices=("benchmark", "pilot", "study", "search"), default="pilot")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--lr-schedule", choices=("adaptive", "fixed"))
    p.add_argument("--entropy-coef", type=float)
    p.add_argument("--run-label", default="")
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
    success_sample = args.success_state_sample
    if args.resume:
        saved_sample = saved_experiment(
            json.loads(args.resume.with_name("experiment.json").read_text())
        ).success_state_sample
        if success_sample is not None and success_sample != saved_sample:
            raise ValueError("Resume success_state_sample differs; use a distinct run")
        success_sample = saved_sample
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
        success_state_sample=success_sample or "current_qpos",
        dynamic_onset_range=args.dynamic_onset_range,
        dynamic_duration_range=args.dynamic_duration_range,
        dynamic_center=args.dynamic_center,
        dynamic_center_jitter=args.dynamic_center_jitter,
        dynamic_travel_range=args.dynamic_travel_range,
        dynamic_panel_half_size=args.dynamic_panel_half_size,
        dynamic_panel_yaw=args.dynamic_panel_yaw,
    )


def optimization_from_args(args):
    """Resolve explicit settings, inheriting saved settings for an exact resume."""
    settings = {"learning_rate": 3e-4, "schedule": "adaptive", "entropy_coef": 0.003}
    if args.resume:
        saved = json.loads(args.resume.with_name("runner.json").read_text())["algorithm"]
        settings.update({key: saved[key] for key in settings})
    for key, value in (
        ("learning_rate", args.learning_rate),
        ("schedule", args.lr_schedule),
        ("entropy_coef", args.entropy_coef),
    ):
        if value is not None:
            if args.resume and value != settings[key]:
                raise ValueError(f"Resume optimization differs in {key}; use a distinct run")
            settings[key] = value
    for key in ("learning_rate", "entropy_coef"):
        value = settings[key]
        if not math.isfinite(value) or value < 0 or (key == "learning_rate" and value == 0):
            raise ValueError(f"Invalid {key}: {value}")
    if args.run_label and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_label):
        raise ValueError("run-label must contain only letters, digits, underscores or hyphens")
    return settings


def restore_training_state(runner, checkpoint, num_envs):
    """Continue after the saved update, retaining the adaptive optimizer LR."""
    runner.load(str(checkpoint))
    # RSL loads Adam state but leaves PPO.learning_rate at its constructor value.
    # The adaptive-KL schedule reads that scalar and would overwrite restored LR.
    rates = {group["lr"] for group in runner.alg.optimizer.param_groups}
    if len(rates) != 1:
        raise ValueError(f"Expected one shared PPO learning rate, found {rates}")
    runner.alg.learning_rate = rates.pop()
    runner.current_learning_iteration += 1
    runner.logger.tot_timesteps = runner.current_learning_iteration * num_envs * 24


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
            if json.dumps(value) != json.dumps(original[key]):
                raise ValueError(f"Resume configuration differs in {key}")
    optimization = optimization_from_args(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment": cfg.to_dict(),
                    "iterations": args.iterations,
                    "optimization": optimization,
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
    if args.run_label:
        name = name.replace(f"_{stamp}", f"_{args.run_label}_{stamp}")
    log_dir = args.resume.parent if args.resume else args.log_root / name
    os.environ.setdefault("WANDB_PROJECT", "active-perception-so101")
    os.environ.setdefault("WANDB_RUN_GROUP", f"plug-{args.job_type}-20261007")
    os.environ.setdefault("WANDB_JOB_TYPE", args.job_type)
    os.environ.setdefault(
        "WANDB_TAGS", f"restored-plug,{args.job_type},{cfg.condition},{cfg.memory}"
    )
    if args.resume:
        archive = log_dir / "resume_history" / stamp
        archive.mkdir(parents=True, exist_ok=False)
        for filename in (
            "experiment.json",
            "runner.json",
            "runtime.json",
            "summary.json",
            "wandb_run.json",
            "iterations.jsonl",
            "gpu.csv",
        ):
            source = log_dir / filename
            if source.exists():
                shutil.copy2(source, archive / filename)
        run_info = json.loads((log_dir / "wandb_run.json").read_text())
        os.environ["WANDB_RUN_ID"] = run_info["id"]
        os.environ["WANDB_RESUME"] = "must"
        os.environ["WANDB_PROJECT"] = run_info["project"]
        os.environ["WANDB_ENTITY"] = run_info["entity"]
    log_dir.mkdir(parents=True, exist_ok=bool(args.resume))
    (log_dir / "experiment.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    runner_options = runner_cfg(cfg, args.iterations)
    for key, value in optimization.items():
        setattr(runner_options.algorithm, key, value)
    runner_options.logger = "wandb"
    runner_options.wandb_project = os.environ["WANDB_PROJECT"]
    runner_options.save_interval = max(1, args.iterations // 2)
    (log_dir / "runner.json").write_text(json.dumps(asdict(runner_options), indent=2))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    metadata = {
        "command": sys.argv,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cudnn": torch.backends.cudnn.version(),
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
    (log_dir / f"runtime_{stamp}.json").write_text(json.dumps(metadata, indent=2))
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
            restore_training_state(runner, args.resume, cfg.num_envs)
            metadata["resume_next_iteration_index"] = runner.current_learning_iteration
            metadata["resume_cumulative_transitions"] = runner.logger.tot_timesteps
            metadata["resume_learning_rate"] = runner.alg.learning_rate
            metadata["resume_state_note"] = (
                "Model/Adam/iteration/common_step_counter restored; simulator episode "
                "states, RNG and GRU hidden state are reset, not saved by native checkpoints."
            )
            for name in ("runtime.json", f"runtime_{stamp}.json"):
                (log_dir / name).write_text(json.dumps(metadata, indent=2))
            print(json.dumps({"resume": metadata}), flush=True)
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
