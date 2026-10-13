"""Verify completed study budgets, immutable sources, selection and physical outcomes."""

import hashlib
import json
from pathlib import Path

from summarize_common_control_study import read_evaluation


def main():
    root = Path(__file__).resolve().parents[1] / "artifacts/common_control_study"
    cases = []
    verified_sources = 0
    evaluation_episodes = 0
    evaluations = 0
    for path in sorted(root.glob("*/plan.json")):
        if not (path.parent / "training_submission.json").exists():
            continue
        plan = json.loads(path.read_text())
        for relative, expected in plan["source_sha256"].items():
            source = Path(plan["source"]) / relative
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, source
            verified_sources += 1
        assert json.loads((path.parent / "gpu_gate/summary.json").read_text())["passed"]
        for case in plan["cases"]:
            directory = path.parent / case["name"]
            completed = json.loads((directory / "completed.json").read_text())
            training = json.loads((directory / "training.json").read_text())
            assert training["actual_transitions"] == plan["transitions_per_policy"]
            validation = []
            for result in directory.glob("validation-*.json"):
                data = read_evaluation(result)
                assert data["seed"] == plan["validation_seed"]
                assert data["split"] == "validation"
                iteration = int(result.stem.rsplit("_", 1)[1])
                validation.append((data["successes"], -iteration, data["checkpoint"]))
                evaluations += 1
                evaluation_episodes += data["episodes"]
            best = max(validation)[2]
            selection = json.loads((directory / "selection.json").read_text())
            assert selection["checkpoint"] == best == completed["selected_checkpoint"]
            reports = list(directory.glob("test-*.json")) + list(
                directory.glob("intervention-*.json")
            )
            expected_count = (
                5
                if plan["stage"] == "camera"
                and case["prepared"]["experiment"]["condition"] == "active"
                else 2
            )
            assert len(reports) == expected_count, directory
            initial = {}
            for result in reports:
                data = read_evaluation(result)
                assert data["checkpoint"] == best
                assert data["seed"] == plan["test_seed"]
                assert data["split"] == "test" and data["episodes"] == 512
                mode = data["experiment"]["occlusion"]
                hashes = data["initial_physics_state_sha256"]
                assert hashes == initial.setdefault(mode, hashes)
                evaluations += 1
                evaluation_episodes += data["episodes"]
            cases.append(
                {
                    "stage": path.parent.name,
                    "case": case["name"],
                    "transitions": training["actual_transitions"],
                    "checkpoint": best,
                    "checkpoint_sha256": hashlib.sha256(Path(best).read_bytes()).hexdigest(),
                }
            )
    assert json.loads((root / "push_v2_gate/summary.json").read_text())["passed"]
    summary = {
        "passed": True,
        "completed_policies": len(cases),
        "training_transitions": sum(c["transitions"] for c in cases),
        "source_files_verified": verified_sources,
        "evaluations_verified": evaluations,
        "physical_episodes_verified": evaluation_episodes,
        "selection": "Every chosen checkpoint matches the predeclared validation-only rule.",
        "cases": cases,
    }
    (root / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
