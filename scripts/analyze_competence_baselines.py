"""Attribute saved push failures to observable terminal criteria, not assumed causes."""

import json
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
out = root / "artifacts/competence_repair"
out.mkdir(exist_ok=True)
rows = []
for task, pattern in [
    ("plug", "artifacts/plug_insertion_study_v2/*/state.json"),
    ("push", "artifacts/push_rgb_arc120/runs/*/state.json"),
]:
    for path in sorted(root.glob(pattern)):
        state = json.loads(path.read_text())
        weights = torch.load(state["checkpoint"], map_location="cpu", weights_only=True)
        std = weights["actor_state_dict"]["distribution.std_param"].tolist()
        for e in state["evaluations"]:
            if e["split"] != "test":
                continue
            data = json.loads(Path(e["path"]).read_text())
            records = data["termination_audit"]["records"]
            failed = [r for r in records if not r["success"]]
            row = dict(
                task=task,
                run=path.parent.name,
                occlusion=e["occlusion"],
                successes=e["successes"],
                episodes=e["episodes"],
                action_std=std,
                failed_median_error_mm=float(
                    np.median([r["qpos_position_error_m"] * 1000 for r in failed])
                ),
                failed_error_below_5mm=sum(r["qpos_position_error_m"] < 0.005 for r in failed),
            )
            if task == "push":
                row.update(
                    failed_xy_below_5mm=sum(r["push_xy_error"] < 0.005 for r in failed),
                    failed_not_withdrawn=sum(not r["push_withdrawn"] for r in failed),
                    failed_contact=sum(not r["push_contact_free"] for r in failed),
                    failed_speed_above_threshold=sum(r["linear_speed"] >= 0.035 for r in failed),
                    failed_z_above_threshold=sum(r["push_z_error"] >= 0.004 for r in failed),
                )
                row["goal_x_bins"] = []
                for low, high in [(-1, -0.075), (-0.075, -0.025), (-0.025, 0.025), (0.025, 1)]:
                    rs = [r for r in records if low <= r["physical_goal_position"][0] < high]
                    row["goal_x_bins"].append(
                        dict(
                            low=low,
                            high=high,
                            episodes=len(rs),
                            successes=sum(r["success"] for r in rs),
                        )
                    )
            rows.append(row)
(out / "baseline_failure_analysis.json").write_text(json.dumps(rows, indent=2) + "\n")
for row in rows:
    if row["occlusion"] == "clean":
        print(json.dumps(row))
