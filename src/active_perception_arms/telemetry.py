"""W&B and compact local timing records for the pinned RSL-RL learning loop."""

import csv
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch


class GpuSampler:
    """Sample device-wide VRAM (includes Warp) and utilization, once per second."""

    def __init__(self, path):
        self.path = Path(path)
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def start(self):
        self.started = time.perf_counter()
        self.thread.start()
        return self

    def _sample(self):
        with self.path.open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["wall_seconds", "uuid", "used_mib", "total_mib", "utilization_pct"])
            while not self.stop_event.is_set():
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                for line in result.stdout.splitlines():
                    fields = [s.strip() for s in line.split(",")]
                    if len(fields) == 4:
                        row = [
                            time.perf_counter() - self.started,
                            fields[0],
                            *[float(x) for x in fields[1:]],
                        ]
                        self.samples.append(row)
                        writer.writerow(row)
                f.flush()
                self.stop_event.wait(1)

    def metrics(self):
        if not self.samples:
            return {}
        # Slurm exposes the allocated device only; record UUIDs to audit this.
        recent = self.samples[-1]
        return {
            "gpu_memory_mib": recent[2],
            "gpu_utilization_pct": recent[4],
            "gpu_peak_memory_mib": max(row[2] for row in self.samples),
            "gpu_mean_utilization_pct": statistics.mean(row[4] for row in self.samples),
        }

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=25)


def install_logging(runner, cfg, args, log_dir, metadata, init_seconds, sampler, process_start):
    """Retain native RSL logger; attach synchronized phase timings and local records."""
    import wandb

    rows = []
    state = {"scalars": {}, "mark": None, "collect": None, "learn_start": None}
    original_init = runner.logger.init_logging_writer
    original_log = runner.logger.log
    original_returns = runner.alg.compute_returns
    original_update = runner.alg.update
    original_act = runner.alg.act
    original_stop = runner.logger.stop_logging_writer
    step_size = cfg.num_envs * runner.cfg["num_steps_per_env"]

    def init():
        original_init()
        run = wandb.run
        run.config.update(
            {
                "experiment": cfg.to_dict(),
                "runtime": metadata,
                "transition_budget": args.iterations * step_size,
                "fixed_view": args.fixed_view,
                "job_type": args.job_type,
                "environment_initialization_seconds": init_seconds,
            }
        )
        run.define_metric("train/transitions")
        run.define_metric("train/*", step_metric="train/transitions")
        run.define_metric("validation/*", step_metric="train/transitions")
        (log_dir / "wandb_run.json").write_text(
            json.dumps(
                {
                    "id": run.id,
                    "url": run.url,
                    "entity": run.entity,
                    "project": run.project,
                    "mode": run.settings.mode,
                },
                indent=2,
            )
        )
        add_scalar = runner.logger.writer.add_scalar

        def scalar(tag, value, global_step=None, **kwargs):
            tag = "train/" + tag
            value = value.item() if isinstance(value, torch.Tensor) else value
            state["scalars"][tag] = value
            add_scalar(tag, value, global_step, **kwargs)

        runner.logger.writer.add_scalar = scalar
        torch.cuda.synchronize()
        state["mark"] = time.perf_counter()

    def act(obs):
        if state["collect"] is None:
            torch.cuda.synchronize()
            state["collect"] = time.perf_counter()
        return original_act(obs)

    def returns(obs):
        torch.cuda.synchronize()
        state["learn_start"] = time.perf_counter()
        state["rollout_seconds"] = state["learn_start"] - state["collect"]
        return original_returns(obs)

    def update():
        result = original_update()
        torch.cuda.synchronize()
        state["update_seconds"] = time.perf_counter() - state["learn_start"]
        return result

    def log(**kwargs):
        now = time.perf_counter()
        iteration = kwargs["it"]
        elapsed = now - state["mark"]
        row = {
            "iteration": iteration + 1,
            "transitions": (iteration + 1) * step_size,
            "iteration_transitions": step_size,
            "warmup": len(rows) < args.warmup_iterations,
            "rollout_seconds": state["rollout_seconds"],
            "ppo_seconds": state["update_seconds"],
            "iteration_seconds": elapsed,
            "transitions_per_second": step_size / elapsed,
            "wall_seconds": now - process_start,
            "torch_peak_bytes": torch.cuda.max_memory_allocated(),
            **sampler.metrics(),
        }
        wandb.log({"train/" + key: value for key, value in row.items()}, step=iteration)
        kwargs["collect_time"] = row["rollout_seconds"]
        kwargs["learn_time"] = row["ppo_seconds"]
        original_log(**kwargs)
        row.update(state["scalars"])
        rows.append(row)
        with (log_dir / "iterations.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        state["mark"] = now  # includes logger/checkpoint overhead in the next interval
        state["collect"] = None

    def stop():
        total = time.perf_counter() - process_start
        steady = [row for row in rows if not row["warmup"]]
        measured = sum(row["iteration_seconds"] for row in steady)
        transitions = sum(row["iteration_transitions"] for row in steady)
        rate = transitions / measured if measured else None
        report = {
            "experiment": cfg.to_dict(),
            "runtime": metadata,
            "wandb": json.loads((log_dir / "wandb_run.json").read_text()),
            "job_type": args.job_type,
            "training_started": True,
            "environment_initialization_seconds": init_seconds,
            "warmup_iterations": args.warmup_iterations,
            "warmup_seconds": sum(r["iteration_seconds"] for r in rows if r["warmup"]),
            "actual_transitions": sum(r["iteration_transitions"] for r in rows),
            "measured_steady_transitions": transitions,
            "measured_steady_seconds": measured,
            "steady_transitions_per_second": rate,
            "total_wall_seconds": total,
            "gpu_hours": total / 3600,
            "steady_iteration_std_seconds": statistics.stdev(
                [r["iteration_seconds"] for r in steady]
            )
            if len(steady) > 1
            else None,
            "extrapolations": {
                str(b): {"seconds": b / rate, "gpu_hours": b / rate / 3600}
                for b in (1228800, 18432000)
            }
            if rate
            else {},
            "timing_note": "One control step produces N transitions; 24 steps/iteration. "
            "PPO time includes returns. Warmup contains lazy compilation. Extrapolations "
            "exclude startup, evaluation and queue time; short-run variance is not a CI.",
            **sampler.metrics(),
        }
        (log_dir / "summary.json").write_text(json.dumps(report, indent=2))
        if args.result:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(json.dumps(report, indent=2))
        wandb.run.summary.update({"training_summary": report})
        for name in (
            "experiment.json",
            "runner.json",
            "runtime.json",
            "summary.json",
            "iterations.jsonl",
        ):
            wandb.save(str(log_dir / name), base_path=str(log_dir))
        # Native save interval is bounded to <=3 checkpoints for this run.
        if args.job_type == "pilot":
            artifact = wandb.Artifact(f"{wandb.run.id}-checkpoints", type="model")
            for path in log_dir.glob("*.pt"):
                artifact.add_file(str(path))
            wandb.log_artifact(artifact)
        original_stop()

    runner.logger.init_logging_writer = init
    runner.logger.log = log
    runner.logger.stop_logging_writer = stop
    runner.alg.act = act
    runner.alg.compute_returns = returns
    runner.alg.update = update
