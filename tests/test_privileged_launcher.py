"""The diagnostic resumes saved learning and tolerates absent optional capture."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SPEC = importlib.util.spec_from_file_location(
    "privileged_diagnostic",
    Path(__file__).resolve().parents[1] / "scripts/run_privileged_diagnostic.py",
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_resume_evaluation_without_retraining_or_rendering_null_capture(tmp_path, monkeypatch):
    plan = dict(
        root=str(tmp_path),
        source=str(tmp_path),
        output=str(tmp_path / "out"),
        logs=str(tmp_path / "logs"),
        python="python",
        label="test",
        source_sha256={},
        num_envs=2048,
        milestones=[2_000_000, 5_000_000],
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    output = Path(plan["output"]) / "push"
    output.mkdir(parents=True)
    run_dir = Path(plan["logs"]) / "push" / "unique"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment.json").write_text("{}")
    checkpoint = run_dir / "model_40.pt"
    torch.save({"iter": 40}, checkpoint)
    state = dict(
        status="failed",
        task="push",
        updates=41,
        transitions=2015232,
        checkpoint=str(checkpoint),
        checkpoint_sha256=diagnostic.hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        evaluations=[],
        error="optional renderer received null",
    )
    (output / "runtime.json").write_text(json.dumps(state))
    launches = []

    def execute(command, **kwargs):
        launches.append(command)
        if "active_perception_arms.train" in command:
            assert "--resume" in command
            updates = int(command[command.index("--iterations") + 1])
            torch.save({"iter": updates - 1}, run_dir / f"model_{updates - 1}.pt")
        elif "active_perception_arms.evaluate" in command:
            report = dict(
                experiment={"task": "push"},
                successes=0,
                episodes=512,
                outcomes=[False] * 512,
                success_rate=0,
                representative_capture=None,
                termination_audit={
                    "success_hold_violations": 0,
                    "success_push_violations": 0,
                    "records": [
                        {
                            "success": False,
                            "push_xy_error": 0.1,
                            "push_z_error": 0,
                            "push_withdrawn": True,
                            "push_contact_free": True,
                            "linear_speed": 0,
                        }
                    ]
                    * 512,
                },
            )
            Path(command[command.index("--output") + 1]).write_text(json.dumps(report))
        else:
            assert command[1].endswith("build_privileged_report.py")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(diagnostic.subprocess, "run", execute)
    monkeypatch.setattr(
        sys, "argv", ["diagnostic", "--plan", str(path), "--task", "push", "--resume-diagnostic"]
    )
    diagnostic.main()
    trained = [c for c in launches if "active_perception_arms.train" in c]
    assert len(trained) == 1
    assert trained[0][trained[0].index("--iterations") + 1] == "102"
    result = json.loads((output / "runtime.json").read_text())
    assert result["status"] == "completed_diagnostic"
    assert result["transitions"] == 5013504
    assert len(result["evaluations"]) == 4
    assert not result["competence_demonstrated"]
