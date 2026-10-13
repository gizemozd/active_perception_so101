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
    REWARD_PROFILES,
    SUCCESS_STATE_SAMPLES,
    TASKS,
    Experiment,
    saved_experiment,
    static_candidates,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--task-revision", help="Fresh task default; resume inherits the saved revision")
    p.add_argument(
        "--push-visibility-revision", choices=("crossbar_v1", "crossbar_v2", "crossbar_v3")
    )
    p.add_argument("--manip-control", choices=("joint", "cartesian"))
    p.add_argument("--actor-observation", choices=("rgb", "privileged"))
    p.add_argument("--reset-xy-jitter", type=float)
    p.add_argument("--push-target-layout", choices=("square_v1", "arc_v1", "arc_v2"))
    p.add_argument("--push-control-revision", choices=("legacy_v1", "wide_v2"))
    p.add_argument("--push-workspace-revision", choices=("partition_v1", "arc_clear_v2"))
    p.add_argument("--plug-control", choices=("absolute", "delta5mm"))
    p.add_argument("--control-revision", choices=("legacy_v1", "pose_v1"))
    p.add_argument("--translation-control", choices=("absolute", "delta"))
    p.add_argument("--translation-step", type=float)
    p.add_argument("--camera-interval", type=int)
    p.add_argument("--camera-workspace", choices=("local_v1", "inspection_v2"))
    p.add_argument("--camera-start", choices=("home", "inspection_v1"))
    p.add_argument("--action-distribution", choices=("gaussian", "squashed"))
    p.add_argument("--save-interval", type=int, help="Checkpoint interval in PPO updates")
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
        "--initialize-from",
        type=Path,
        help="Fresh optimizer/run using saved actor/critic weights; exploration resets to --initial-std",
    )
    p.add_argument(
        "--reward-profile",
        choices=REWARD_PROFILES,
        help=(
            "Fresh default: plug_insertion for current-qpos plug, transfer_staged for "
            "transfer v2, progress otherwise; resume inherits"
        ),
    )
    p.add_argument(
        "--initial-std",
        type=float,
        help="Positive Gaussian initial std; compact default 0.4, legacy_ff 1.0; resume inherits",
    )
    p.add_argument(
        "--success-state-sample",
        choices=SUCCESS_STATE_SAMPLES,
        help="Fresh default: current_qpos; resume inherits saved sampling",
    )
    p.add_argument("--perturb-push", action="store_true")
    p.add_argument("--memory", choices=("gru", "none"))
    p.add_argument("--policy-profile", choices=("compact", "legacy_ff"))
    p.add_argument(
        "--plug-wrist-yaw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add a bounded fourth manipulator action and physical orientation success check",
    )
    p.add_argument("--job-type", choices=("benchmark", "pilot", "study", "search"), default="pilot")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--lr-schedule", choices=("adaptive", "fixed"))
    p.add_argument("--entropy-coef", type=float)
    p.add_argument("--action-mean-penalty", type=float)
    p.add_argument("--run-label", default="")
    p.add_argument("--warmup-iterations", type=int, default=3)
    p.add_argument("--result", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p


def experiment_from_args(args):
    shared = {}
    for key, default in (
        ("control_revision", "legacy_v1"),
        ("translation_control", "delta"),
        ("translation_step", 0.005),
        ("camera_interval", 1),
        ("camera_workspace", "local_v1"),
        ("camera_start", "home"),
    ):
        value = getattr(args, key)
        if args.resume:
            saved = saved_experiment(
                json.loads(args.resume.with_name("experiment.json").read_text())
            )
            if value is not None and value != getattr(saved, key):
                raise ValueError(f"Resume {key} differs; use a distinct run")
            value = getattr(saved, key)
        shared[key] = default if value is None else value
    if shared["control_revision"] == "pose_v1" and args.fixed_view:
        raise ValueError(
            "pose_v1 uses the same arm-mounted home camera in fixed and active conditions"
        )
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
    reward_profile = args.reward_profile
    profile = policy_profile_from_args(args)
    memory = args.memory or ("none" if profile == "legacy_ff" else "gru")
    wrist_yaw = args.plug_wrist_yaw
    task_revision = args.task_revision
    manip_control = args.manip_control
    actor_observation = args.actor_observation
    reset_xy_jitter = args.reset_xy_jitter
    plug_control = args.plug_control
    push_visibility_revision = args.push_visibility_revision
    push_target_layout = args.push_target_layout
    push_control_revision = args.push_control_revision
    push_workspace_revision = args.push_workspace_revision
    if args.resume:
        saved_cfg = saved_experiment(
            json.loads(args.resume.with_name("experiment.json").read_text())
        )
        saved_sample = saved_cfg.success_state_sample
        if success_sample is not None and success_sample != saved_sample:
            raise ValueError("Resume success_state_sample differs; use a distinct run")
        success_sample = saved_sample
        if reward_profile is not None and reward_profile != saved_cfg.reward_profile:
            raise ValueError("Resume reward_profile differs; use a distinct run")
        reward_profile = saved_cfg.reward_profile
        if args.memory is None:
            memory = saved_cfg.memory
        if wrist_yaw is not None and wrist_yaw != saved_cfg.plug_wrist_yaw:
            raise ValueError("Resume plug_wrist_yaw differs; use a distinct run")
        wrist_yaw = saved_cfg.plug_wrist_yaw
        if task_revision is not None and task_revision != saved_cfg.task_revision:
            raise ValueError("Resume task_revision differs; use a distinct run")
        task_revision = saved_cfg.task_revision
        for name, value in (("manip_control", manip_control), ("reset_xy_jitter", reset_xy_jitter)):
            if value is not None and value != getattr(saved_cfg, name):
                raise ValueError(f"Resume {name} differs; use a distinct run")
        manip_control = saved_cfg.manip_control
        if actor_observation is not None and actor_observation != saved_cfg.actor_observation:
            raise ValueError("Resume actor_observation differs; use a distinct run")
        actor_observation = saved_cfg.actor_observation
        reset_xy_jitter = saved_cfg.reset_xy_jitter
        if (
            push_visibility_revision is not None
            and push_visibility_revision != saved_cfg.push_visibility_revision
        ):
            raise ValueError("Resume push_visibility_revision differs; use a distinct run")
        push_visibility_revision = saved_cfg.push_visibility_revision
        if push_target_layout is not None and push_target_layout != saved_cfg.push_target_layout:
            raise ValueError("Resume push_target_layout differs; use a distinct run")
        push_target_layout = saved_cfg.push_target_layout
        if (
            push_control_revision is not None
            and push_control_revision != saved_cfg.push_control_revision
        ):
            raise ValueError("Resume push_control_revision differs; use a distinct run")
        push_control_revision = saved_cfg.push_control_revision
        if (
            push_workspace_revision is not None
            and push_workspace_revision != saved_cfg.push_workspace_revision
        ):
            raise ValueError("Resume push_workspace_revision differs; use a distinct run")
        push_workspace_revision = saved_cfg.push_workspace_revision
        if plug_control is not None and plug_control != saved_cfg.plug_control:
            raise ValueError("Resume plug_control differs; use a distinct run")
        plug_control = saved_cfg.plug_control
    return Experiment(
        **shared,
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
        memory=memory,
        policy_profile=profile,
        plug_wrist_yaw=bool(wrist_yaw),
        plug_control=plug_control or "absolute",
        task_revision=task_revision,
        manip_control=manip_control or "joint",
        actor_observation=actor_observation or "rgb",
        reset_xy_jitter=0.008 if reset_xy_jitter is None else reset_xy_jitter,
        success_state_sample=success_sample or "current_qpos",
        reward_profile=reward_profile,
        push_visibility_revision=push_visibility_revision,
        push_target_layout=push_target_layout,
        push_control_revision=push_control_revision or "legacy_v1",
        push_workspace_revision=push_workspace_revision or "partition_v1",
        dynamic_onset_range=args.dynamic_onset_range,
        dynamic_duration_range=args.dynamic_duration_range,
        dynamic_center=args.dynamic_center,
        dynamic_center_jitter=args.dynamic_center_jitter,
        dynamic_travel_range=args.dynamic_travel_range,
        dynamic_panel_half_size=args.dynamic_panel_half_size,
        dynamic_panel_yaw=args.dynamic_panel_yaw,
    )


def policy_profile_from_args(args):
    profile = args.policy_profile
    if args.resume:
        path = args.resume.with_name("experiment.json")
        # Some unit-level optimizer callers only provide runner.json.
        saved = (
            saved_experiment(json.loads(path.read_text())).policy_profile
            if path.exists()
            else "compact"
        )
        if profile is not None and profile != saved:
            raise ValueError("Resume policy_profile differs; use a distinct run")
        profile = saved
    return profile or "compact"


def optimization_from_args(args):
    """Resolve explicit settings, inheriting saved settings for an exact resume."""
    settings = {"learning_rate": 3e-4, "schedule": "adaptive", "entropy_coef": 0.003}
    if policy_profile_from_args(args) == "legacy_ff":
        settings.update(learning_rate=0.001, entropy_coef=0.005)
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


def initial_std_from_args(args):
    initial_std = 1.0 if policy_profile_from_args(args) == "legacy_ff" else 0.4
    if args.resume:
        runner = json.loads(args.resume.with_name("runner.json").read_text())
        initial_std = float(runner["actor"]["distribution_cfg"]["init_std"])
    if args.initial_std is not None:
        if args.resume and args.initial_std != initial_std:
            raise ValueError("Resume initial_std differs; use a distinct run")
        initial_std = args.initial_std
    if not math.isfinite(initial_std) or initial_std <= 0:
        raise ValueError("initial_std must be positive and finite")
    return initial_std


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
    if args.resume and args.initialize_from:
        raise ValueError("Use either resume or initialize-from")
    if args.resume:
        original = saved_experiment(
            json.loads(args.resume.with_name("experiment.json").read_text())
        ).to_dict()
        # A resume restores the same experiment; interventions belong in evaluation.
        for key, value in cfg.to_dict().items():
            if json.dumps(value) != json.dumps(original[key]):
                raise ValueError(f"Resume configuration differs in {key}")
    optimization = optimization_from_args(args)
    initial_std = initial_std_from_args(args)
    distribution = args.action_distribution
    if args.resume:
        saved_class = json.loads(args.resume.with_name("runner.json").read_text())["actor"][
            "distribution_cfg"
        ].get("class_name", "GaussianDistribution")
        saved_kind = "squashed" if saved_class.endswith(":SquashedGaussian") else "gaussian"
        if distribution is not None and distribution != saved_kind:
            raise ValueError("Resume action distribution differs")
        distribution = saved_kind
    distribution = distribution or "gaussian"
    mean_penalty = args.action_mean_penalty
    if args.resume:
        saved_distribution = json.loads(args.resume.with_name("runner.json").read_text())["actor"][
            "distribution_cfg"
        ]
        saved_penalty = saved_distribution.get("mean_penalty", 0.0)
        if mean_penalty is not None and mean_penalty != saved_penalty:
            raise ValueError("Resume action-mean-penalty differs; use a distinct run")
        mean_penalty = saved_penalty
    mean_penalty = mean_penalty or 0.0
    if mean_penalty < 0 or not math.isfinite(mean_penalty):
        raise ValueError("action-mean-penalty must be finite and nonnegative")
    if mean_penalty and distribution == "squashed":
        raise ValueError("Squashed policy already bounds latent means; do not add mean penalty")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment": cfg.to_dict(),
                    "iterations": args.iterations,
                    "optimization": optimization,
                    "initial_std": initial_std,
                    "action_distribution": distribution,
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
    runner_options.actor.distribution_cfg["init_std"] = initial_std
    if distribution == "squashed":
        runner_options.actor.distribution_cfg.update(
            class_name="active_perception_arms.bounded_distribution:SquashedGaussian",
            std_range=(0.05, 1.0),
        )
    if mean_penalty:
        runner_options.actor.distribution_cfg.update(
            class_name="active_perception_arms.bounded_distribution:RegularizedGaussian",
            mean_penalty=mean_penalty,
            std_range=(0.03, 0.5),
        )
    for key, value in optimization.items():
        setattr(runner_options.algorithm, key, value)
    runner_options.logger = "wandb"
    runner_options.wandb_project = os.environ["WANDB_PROJECT"]
    runner_options.save_interval = args.save_interval or max(1, args.iterations // 2)
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
        "initialize_from": str(args.initialize_from) if args.initialize_from else None,
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
        if args.initialize_from:
            import hashlib

            with args.initialize_from.open("rb") as stream:
                metadata["initialize_from_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
            weights = torch.load(args.initialize_from, map_location=args.device, weights_only=True)
            runner.alg.actor.load_state_dict(weights["actor_state_dict"])
            runner.alg.critic.load_state_dict(weights["critic_state_dict"])
            with torch.no_grad():
                source_cfg = saved_experiment(
                    json.loads(args.initialize_from.with_name("experiment.json").read_text())
                )
                if (
                    cfg.task == "push"
                    and source_cfg.push_control_revision != cfg.push_control_revision
                ):
                    from .manipulation import bounds

                    old_low, old_high = [
                        torch.tensor(v, device=args.device)
                        for v in bounds("push", source_cfg.push_control_revision)
                    ]
                    new_low, new_high = [
                        torch.tensor(v, device=args.device)
                        for v in bounds("push", cfg.push_control_revision)
                    ]
                    scale = (old_high - old_low) / (new_high - new_low)
                    offset = (old_high + old_low - new_high - new_low) / (new_high - new_low)
                    # Preserve the old actor's physical (unclipped) TCP requests while
                    # making the newly reachable portion of the workspace available.
                    head = [
                        m for m in runner.alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)
                    ][-1]
                    head.weight[:3] *= scale[:, None]
                    head.bias[:3] = head.bias[:3] * scale + offset
                    metadata["action_bound_remap"] = {
                        "scale": scale.tolist(),
                        "offset": offset.tolist(),
                    }
                runner.alg.actor.distribution.std_param.fill_(initial_std)
            (log_dir / "runtime.json").write_text(json.dumps(metadata, indent=2))
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
