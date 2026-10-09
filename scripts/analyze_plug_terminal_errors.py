"""Summarize physical errors at the first terminal step of existing plug evaluations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def collect(root):
    reports = []
    analysis = root / "artifacts/plug_analysis"
    groups = (
        ("original_checkpoints", analysis.glob("corrected-*-s0.json")),
        (
            "historical_tuning_checkpoints",
            (root / "artifacts/hparam_search/continuation").glob(
                "*/corrected-evaluation-repeat*.json"
            ),
        ),
        ("corrected_reward_screen", (analysis / "reward_repair").glob("*/evaluation-repeat*.json")),
        (
            "corrected_reward_continuation",
            (analysis / "reward_repair/continuation").glob("*/evaluation-repeat*.json"),
        ),
        ("fresh_exploration", (analysis / "exploration_repair").glob("*/evaluation-repeat*.json")),
        (
            "camera_control_diagnostics",
            (analysis / "camera_controls").glob("*/evaluation-repeat*.json"),
        ),
    )
    for stage, paths in groups:
        for path in sorted(paths):
            data = json.loads(path.read_text())
            assert data["experiment"]["success_state_sample"] == "current_qpos"
            audit = data["termination_audit"]
            records = audit["records"]
            assert len(records) == len(data["variants"]) == len(data["outcomes"]) == 512
            assert sorted(record["episode"] for record in records) == list(range(512))
            assert all(
                audit[key] == 0
                for key in (
                    "success_hold_violations",
                    "success_distance_violations",
                    "success_three_sample_violations",
                    "success_qpos_above_distance_threshold",
                )
            )
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            for variant in ("xm", "xp", "ym", "yp"):
                selected = [
                    record for record in records if data["variants"][record["episode"]] == variant
                ]
                error = np.array([record["qpos_position_error_m"] for record in selected]) * 1000
                success = np.array([bool(record["success"]) for record in selected])
                assert len(selected) == 128 and np.isfinite(error).all()
                assert int(success.sum()) == data["per_variant"][variant]["successes"]
                reports.append(
                    {
                        "stage": stage,
                        "case": data["experiment"]["condition"]
                        if stage == "original_checkpoints"
                        else path.parent.name,
                        "execution": path.stem,
                        "source_report": str(path.relative_to(root)),
                        "source_report_sha256": digest,
                        "checkpoint": data["checkpoint"],
                        "training_seed": data["training_seed"],
                        "variant": variant,
                        "episodes": 128,
                        "successes": int(success.sum()),
                        "success_rate": float(success.mean()),
                        "mean_terminal_error_mm": float(error.mean()),
                        "median_terminal_error_mm": float(np.median(error)),
                        "p25_terminal_error_mm": float(np.quantile(error, 0.25)),
                        "p75_terminal_error_mm": float(np.quantile(error, 0.75)),
                        "p90_terminal_error_mm": float(np.quantile(error, 0.90)),
                        "mean_unsuccessful_terminal_error_mm": float(error[~success].mean())
                        if (~success).any()
                        else None,
                        "terminal_below_2mm_without_success": int(((error < 2) & ~success).sum()),
                        "terminal_below_5mm_fraction": float((error < 5).mean()),
                        "terminal_below_10mm_fraction": float((error < 10).mean()),
                        "terminal_below_20mm_fraction": float((error < 20).mean()),
                        "physical_criterion_violations": 0,
                    }
                )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    rows = collect(root)
    output = root / "artifacts/plug_analysis"
    with (output / "physical_terminal_errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "scope": "Raw physical object-origin error at the FIRST terminal step of existing scored evaluations. Success is held-three, while near-goal fractions describe a single terminal sample.",
        "limitations": "No trajectory minima or continuous dwell inferred. Numeric repeats are separate executions of one training seed. Errors do not distinguish perception, IK, actuator tracking or contact causes. No policy evaluation or physics step is run.",
        "units": "Distances are millimeters; CSV rates and fractions are in [0,1].",
        "rows": rows,
    }
    (output / "physical_terminal_errors.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Summarized {len(rows)} variant rows from {len(rows) // 4} physical evaluations")


if __name__ == "__main__":
    main()
