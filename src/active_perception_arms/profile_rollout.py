"""Disposable phase attribution; CUDA event spans include host scheduling gaps.

Normal timing is measured before instrumentation. Instrumented phase times are
diagnostic, not kernel-only utilization or end-to-end training throughput.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import wandb
from rsl_rl.utils import resolve_callable

from .config import Experiment
from .environment import make_env
from .policy import runner_cfg
from .telemetry import GpuSampler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    cfg = Experiment(task="plug", condition="active", num_envs=args.num_envs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "rollout_phase_profile_no_learning",
        "experiment": cfg.to_dict(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "steps_per_phase": args.steps,
        "units": "N environment transitions per vector control step; 20 physics substeps",
    }
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "active-perception-so101"),
        group="plug-scaling-20261007",
        job_type="benchmark-profile",
        tags=["disposable", "profile", "active", "gru"],
        name=f"plug_active_v0_gru_s0_n{cfg.num_envs}_phase_profile",
        config=report,
    )
    report["wandb_url"] = run.url
    sampler = GpuSampler(args.output.with_suffix(".gpu.csv")).start()
    start = time.perf_counter()
    env = make_env(cfg, "cuda:0")
    torch.cuda.synchronize()
    report["environment_initialization_seconds"] = time.perf_counter() - start
    originals = []
    samples = defaultdict(list)
    try:
        obs, _ = env.reset()
        rc = asdict(runner_cfg(cfg))
        options = rc["actor"]
        actor = (
            resolve_callable(options.pop("class_name"))(
                obs, rc["obs_groups"], "actor", cfg.action_dim, **options
            )
            .to("cuda:0")
            .eval()
        )

        def wrap(obj, method, label):
            original = getattr(obj, method)
            originals.append((obj, method, original))

            def timed(*args, **kwargs):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                host = time.perf_counter()
                result = original(*args, **kwargs)
                host = time.perf_counter() - host
                end.record()
                samples[label].append((begin, end, host))
                return result

            setattr(obj, method, timed)

        with torch.inference_mode():
            start = time.perf_counter()
            for _ in range(25):
                obs, _, term, trunc, _ = env.step(actor(obs))
                actor.reset(term | trunc)
            torch.cuda.synchronize()
            report["warmup_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            for _ in range(args.steps):
                obs, _, term, trunc, _ = env.step(actor(obs))
                actor.reset(term | trunc)
            torch.cuda.synchronize()
            report["baseline_seconds"] = time.perf_counter() - start
            report["baseline_transitions"] = cfg.num_envs * args.steps
            report["baseline_transitions_per_second"] = (
                report["baseline_transitions"] / report["baseline_seconds"]
            )
            for obj, method, label in (
                (actor, "forward", "actor"),
                (env.action_manager, "process_action", "action_processing_ik"),
                (env.action_manager, "apply_action", "apply_targets"),
                (env.scene, "write_data_to_sim", "write_sim"),
                (env.scene, "update", "scene_update"),
                (env.sim, "step", "physics"),
                (env.sim, "forward", "forward"),
                (env.sim, "sense", "camera_rendering"),
                (env.observation_manager, "compute", "observations"),
                (env.reward_manager, "compute", "rewards"),
                (env.termination_manager, "compute", "terminations"),
                (env, "_reset_idx", "reset_inclusive"),
                (env.action_manager.get_term("arms").plug_ik, "reset_manip", "reset_ik_nested"),
            ):
                wrap(obj, method, label)
            start = time.perf_counter()
            for _ in range(args.steps):
                obs, _, term, trunc, _ = env.step(actor(obs))
                actor.reset(term | trunc)
            torch.cuda.synchronize()
            report["instrumented_seconds"] = time.perf_counter() - start
        report["phases"] = {
            label: {
                "calls": len(values),
                "host_dispatch_ms_per_control_step": sum(v[2] for v in values) * 1000 / args.steps,
                "cuda_stream_span_ms_per_control_step": sum(
                    begin.elapsed_time(end) for begin, end, _ in values
                )
                / args.steps,
            }
            for label, values in samples.items()
        }
        report["gpu_metrics"] = sampler.metrics()
        report["note"] = (
            "CUDA event spans include host launch gaps, not only GPU kernel work. "
            "Instrumentation adds overhead. reset_ik_nested overlaps reset_inclusive. "
            "Baseline and instrumented windows are successive, not identical trajectories. "
            "No PPO optimization; compare real training benchmarks separately."
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        run.config.update(report)
        run.log(
            {
                "benchmark/transitions_per_second": report["baseline_transitions_per_second"],
                **{
                    f"profile/{name}_ms": values["cuda_stream_span_ms_per_control_step"]
                    for name, values in report["phases"].items()
                },
            }
        )
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for obj, method, original in originals:
            setattr(obj, method, original)
        sampler.close()
        env.close()
        run.finish()


if __name__ == "__main__":
    main()
