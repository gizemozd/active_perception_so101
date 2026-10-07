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
order = {c: i for i, c in enumerate(("wrist", "wrist_static", "initial", "active"))}
rows.sort(key=lambda r: order[r["condition"]])
with (a.root / "pilot_comparison.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
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
        f"- {r['condition']}: [training]({r['training_wandb']}), [original validation]({r['validation_wandb']}), [JSON]({Path(r['source']).name})."
    )
lines += [
    "",
    "## Recorded diagnostic repeats and media",
    "",
    "Separate replay did not reproduce every rare success. Original scores above are preserved.",
    "A separately labeled repeat used identical seeds/batch size and captured actual state/RGB during evaluation; videos render these saved trajectories without resimulation.",
    "",
    "| Condition | Original successes | Recorded-repeat successes | Changed episode outcomes | Videos |",
    "|---|---:|---:|---:|---|",
]
for r in rows:
    repeat_path = a.root / f"capture-evaluation-{r['condition']}-s0.json"
    if not repeat_path.exists():
        continue
    repeat = json.loads(repeat_path.read_text())
    original = json.loads(Path(r["source"]).read_text())
    changed = sum(x != y for x, y in zip(repeat["outcomes"], original["outcomes"], strict=True))
    lines.append(
        f"| {r['condition']} | {r['successes']} | {repeat['successes']} | {changed} | [recorded videos]({repeat['wandb_url']}) |"
    )
lines += [
    "",
    "Repeatability of individual outcomes is unresolved; do not use these small score differences to rank sensing conditions.",
    "Initial-only has no successes in its recorded repeat. Its verified original episode-213 success clip is retained in the original validation W&B run; batch 1 passed replay assertions before the later batch-2 failure.",
    "",
    "[Learning curves](learning_curves.png) / [CSV](learning_curves.csv), [representative frames](recorded_representative_frames.png), [media checks](media_validation.json), [trajectory diagnostics](trajectory_diagnostics.json).",
    "Native training success curves average reset batches and carry the last value between resets; they are not overall episode-weighted success rates.",
]
(a.root / "PILOTS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
