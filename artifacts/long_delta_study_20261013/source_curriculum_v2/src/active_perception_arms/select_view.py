"""Select one searched fixed view on validation performance, never test results."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .config import static_candidates


def select(reports, expected_seeds=3, allow_incomplete=False):
    groups = defaultdict(list)
    identity = None
    seen = set()
    for path in reports:
        report = json.loads(Path(path).read_text())
        exp = report["experiment"]
        if report["split"] != "validation":
            raise ValueError("View selection accepts validation results only")
        if exp["condition"] not in ("static", "wrist_static"):
            raise ValueError("Expected fixed-view policies")
        if any(
            value is not None and value is not False for value in report["interventions"].values()
        ):
            raise ValueError("Do not select views from intervention evaluations")
        key = (
            exp["task"],
            exp["condition"],
            exp["occlusion"],
            exp["memory"],
            exp["width"],
            exp["height"],
            exp.get("task_revision"),
            exp.get("episode_seconds"),
            exp.get("initial_seconds"),
            tuple(exp["fixed_lookat"]),
            exp.get("plug_variant"),
            report["seed"],
            report["episodes"],
        )
        if identity is None:
            identity = key
        if identity != key:
            raise ValueError(
                "Mixing tasks, conditions, sensors, or evaluation protocols is invalid"
            )
        position = tuple(exp["fixed_position"])
        candidates = static_candidates(exp["task"])
        if position not in candidates:
            raise ValueError("View is outside the published search grid")
        index = candidates.index(position)
        seed = report["training_seed"]
        if (index, seed) in seen:
            raise ValueError(
                "Duplicate view/training seed; choose one checkpoint per run before ranking"
            )
        seen.add((index, seed))
        groups[index].append(report)
    if not groups:
        raise ValueError("No validation reports")
    seed_sets = {index: {r["training_seed"] for r in rows} for index, rows in groups.items()}
    if (
        any(len(seeds) != expected_seeds for seeds in seed_sets.values())
        or len({tuple(sorted(s)) for s in seed_sets.values()}) != 1
    ):
        raise ValueError("Every view requires the same training seeds")
    if not allow_incomplete and len(groups) != len(static_candidates()):
        raise ValueError(
            "All 26 searched viewpoints are required; use --allow-incomplete for a labeled pilot"
        )
    scores = {i: sum(r["success_rate"] for r in rows) / len(rows) for i, rows in groups.items()}
    winner = min(scores, key=lambda i: (-scores[i], i))
    return {
        "task": identity[0],
        "condition": identity[1],
        "occlusion": identity[2],
        "selected_view": winner,
        "position": static_candidates(identity[0])[winner],
        "validation_scores": scores,
        "training_seeds": sorted(seed_sets[winner]),
        "complete_grid": len(groups) == len(static_candidates()),
        "selection_rule": "Mean validation success across training seeds; ties choose lower index.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/selected_view.json"))
    args = parser.parse_args(argv)
    result = select(args.reports, args.expected_seeds, args.allow_incomplete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
