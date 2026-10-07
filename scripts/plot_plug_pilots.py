"""Export compact seed-0 learning curves from local logs, independently of W&B."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--logs", type=Path, default=Path("logs/pilots"))
p.add_argument("--output", type=Path, default=Path("artifacts/cluster_pilot"))
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
rows = []
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for directory in sorted(a.logs.iterdir()):
    if not (directory / "iterations.jsonl").exists():
        continue
    cfg = json.loads((directory / "experiment.json").read_text())
    history = [
        json.loads(line) for line in (directory / "iterations.jsonl").read_text().splitlines()
    ]
    success_keys = {
        key for r in history for key in r if key.startswith("train/") and "success_rate" in key
    }
    assert len(success_keys) == 1, success_keys
    success_key = next(iter(success_keys))
    for r in history:
        rows.append(
            {
                "condition": cfg["condition"],
                "seed": cfg["seed"],
                "transitions": r["transitions"],
                "iteration": r["iteration"],
                "reward": r.get("train/Train/mean_reward"),
                "success_rate": r.get(success_key),
                "episode_length_steps": r.get("train/Train/mean_episode_length"),
                "rollout_seconds": r["rollout_seconds"],
                "ppo_seconds": r["ppo_seconds"],
                "iteration_seconds": r["iteration_seconds"],
            }
        )
    for ax, key in zip(axes, ("train/Train/mean_reward", success_key), strict=True):
        subset = [r for r in history if key in r]
        ax.plot(
            [r["transitions"] for r in subset], [r[key] for r in subset], label=cfg["condition"]
        )
axes[0].set_ylabel("Training episode reward")
axes[1].set_ylabel("Training success metric")
axes[1].set_xlabel("Cumulative environment transitions")
axes[1].set_ylim(-0.02, 1.02)
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend()
fig.suptitle("Restored plug — exploratory seed-0 pilots (training metrics)")
fig.tight_layout()
fig.savefig(a.output / "learning_curves.png", dpi=180)
fig.savefig(a.output / "learning_curves.pdf")
with (a.output / "learning_curves.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Exported {len(rows)} iteration records")
