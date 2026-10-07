"""Compare compact measured inference/PPO reports; never infer training from inference."""

import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, default=Path("artifacts/cluster_pilot"))
a = p.parse_args()
rows = []
for path in sorted(a.root.glob("bench-*-ppo.json")):
    r = json.loads(path.read_text())
    e = r["experiment"]
    rows.append(
        {
            "condition": e["condition"],
            "num_envs": e["num_envs"],
            "transitions_per_second": r["steady_transitions_per_second"],
            "rollout_seconds": r["mean_rollout_seconds"],
            "ppo_seconds": r["mean_ppo_seconds"],
            "iteration_seconds": r["mean_iteration_seconds"],
            "initialization_seconds": r["environment_initialization_seconds"],
            "warmup_seconds": r["warmup_seconds"],
            "steady_transitions": r["measured_steady_transitions"],
            "steady_wall_seconds": r["measured_steady_seconds"],
            "peak_gpu_mib": r.get("gpu_peak_memory_mib"),
            "mean_gpu_utilization_pct": r.get("gpu_mean_utilization_pct"),
            "pilot_gpu_hours": r["extrapolations"]["1228800"]["gpu_hours"],
            "full_gpu_hours": r["extrapolations"]["18432000"]["gpu_hours"],
            "slurm_job_id": r["runtime"]["slurm_job_id"],
            "node": r["runtime"]["hostname"],
            "wandb_url": r["wandb"]["url"],
            "source": str(path),
        }
    )
assert rows, "No measured PPO summaries"
order = {condition: i for i, condition in enumerate(("wrist", "wrist_static", "initial", "active"))}
rows.sort(key=lambda r: (order[r["condition"]], r["num_envs"]))
with (a.root / "benchmark_comparison.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
lines = [
    "# Measured PPO comparison",
    "",
    "Disposable runs; first three iterations excluded as warmup.",
    "One vectorized control step gives N environment transitions; 24 steps/iteration.",
    "Pilot/full GPU-hours are extrapolations, excluding setup, evaluation and queues.",
    "",
    "| Condition | N | transitions/s | Rollout s | PPO s | Iteration s | Peak MiB | Pilot GPU-h | Full GPU-h |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['condition']} | {r['num_envs']} | {r['transitions_per_second']:.1f} | "
        f"{r['rollout_seconds']:.3f} | {r['ppo_seconds']:.3f} | {r['iteration_seconds']:.3f} | "
        f"{r['peak_gpu_mib']} | {r['pilot_gpu_hours']:.3f} | {r['full_gpu_hours']:.3f} |"
    )
(a.root / "BENCHMARKS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
