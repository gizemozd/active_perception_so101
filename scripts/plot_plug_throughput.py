"""Plot measured PPO scaling and separately instrumented rollout phase spans."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("artifacts/cluster_pilot")
rows = list(csv.DictReader((root / "benchmark_comparison.csv").open()))
profile = json.loads((root / "profile-active-512.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for condition in ("wrist", "wrist_static", "initial", "active"):
    selected = [r for r in rows if r["condition"] == condition]
    axes[0].plot(
        [int(r["num_envs"]) for r in selected],
        [float(r["transitions_per_second"]) / 1000 for r in selected],
        marker="o",
        label=condition,
    )
axes[0].set_xscale("log", base=2)
axes[0].set_xticks([64, 128, 256, 512, 1024, 2048, 4096])
axes[0].set_xticklabels([64, 128, 256, 512, 1024, 2048, 4096], rotation=35)
axes[0].set_xlabel("Parallel environments on one GPU")
axes[0].set_ylabel("Training transitions/s (thousands)")
axes[0].set_title("Full PPO: rollout + optimization\n12 iterations, first 3 excluded")
axes[0].legend(frameon=False)
axes[0].grid(alpha=0.2)
phases = {
    "physics": "Physics (20 substeps)",
    "camera_rendering": "Camera rendering",
    "forward": "Physics forward",
    "action_processing_ik": "Action / IK",
    "actor": "Actor inference",
    "reset_inclusive": "Reset (amortized)",
}
axes[1].barh(
    list(phases.values()),
    [profile["phases"][p]["cuda_stream_span_ms_per_control_step"] for p in phases],
    color="#5479a4",
)
axes[1].invert_yaxis()
axes[1].set_xlabel("CUDA stream span, ms per vector control step")
axes[1].set_title("Active / N512 rollout attribution\nInstrumented; excludes PPO updates")
axes[1].grid(axis="x", alpha=0.2)
fig.suptitle("Restored plug · 128×96 RGB · GRU · RTX PRO 6000 Blackwell")
fig.tight_layout()
fig.savefig(root / "throughput_scaling.png", dpi=180)
fig.savefig(root / "throughput_scaling.pdf")
