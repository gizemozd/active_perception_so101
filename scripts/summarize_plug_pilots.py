"""Export exploratory pilot results; preserve failures and distinguish training/validation."""

import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, default=Path("artifacts/cluster_pilot"))
a = p.parse_args()
rows = []
for path in sorted(a.root.glob("evaluation-*.json")):
    r = json.loads(path.read_text())
    if r["interventions"] != dict(
        freeze_camera_after=None, hold_external_after=None, reset_memory=False, camera_trace_in=None
    ):
        continue
    d = Path(r["checkpoint"]).parent
    training = json.loads((d / "summary.json").read_text())
    e = r["experiment"]
    rows.append(
        dict(
            condition=e["condition"],
            seed=r["training_seed"],
            episodes=r["episodes"],
            successes=r["successes"],
            success_rate=r["success_rate"],
            **{v + "_success": x["success_rate"] for v, x in r["per_variant"].items()},
            mean_episode_seconds=r["mean_episode_seconds"],
            success_completion_seconds=r["mean_success_completion_seconds"],
            camera_joint_travel_rad=r["mean_camera_joint_travel_rad"],
            training_transitions=training["final_cumulative_transitions"],
            training_seconds=training.get(
                "wall_seconds_including_wandb_finish", training["total_wall_seconds"]
            ),
            training_gpu_hours=training.get(
                "gpu_hours_including_wandb_finish", training["gpu_hours"]
            ),
            training_wandb=training["wandb"]["url"],
            validation_wandb=r.get("wandb_url"),
            checkpoint=r["checkpoint"],
            source=str(path),
        )
    )
assert rows, "No completed pilot evaluations"
with (a.root / "pilot_comparison.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# Exploratory seed-0 plug pilots",
    "",
    "512 matched validation episodes per policy, 128 per variant, batch 128, reset seeds 10000–10003.",
    "Each trained for 1,228,800 transitions. View 7 is a diagnostic candidate, not a searched optimum.",
    "One training seed supports feasibility and diagnosis, not a statistically supported condition ranking.",
    "",
    "| Condition | Success /512 | xm | xp | ym | yp | Success completion s | Camera travel rad | Training GPU-h |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    completion = (
        "n/a"
        if r["success_completion_seconds"] is None
        else f"{r['success_completion_seconds']:.2f}"
    )
    lines.append(
        f"| {r['condition']} | {r['successes']} ({r['success_rate']:.1%}) | "
        + " | ".join(f"{r[v + '_success']:.1%}" for v in ("xm", "xp", "ym", "yp"))
        + f" | {completion} | {r['camera_joint_travel_rad']:.2f} | {r['training_gpu_hours']:.3f} |"
    )
lines += [
    "",
    "Camera travel is summed absolute camera-arm joint motion; terminal reset jumps are excluded.",
    "Training GPU-hours are measured application wall time on one GPU, including W&B finish; Slurm allocation time is recorded separately.",
    "",
]
for r in rows:
    lines.append(
        f"- {r['condition']}: [training]({r['training_wandb']}), [validation/videos]({r['validation_wandb']}), [JSON]({Path(r['source']).name})."
    )
(a.root / "PILOTS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
