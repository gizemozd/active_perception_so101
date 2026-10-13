"""Budget accounting and durable submission behavior without Slurm/learning."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "manipulation_study", Path(__file__).resolve().parents[1] / "scripts/run_manipulation_study.py"
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def plan(tmp_path):
    return dict(
        root=str(tmp_path),
        source=str(tmp_path / "source"),
        output=str(tmp_path / "out"),
        python="python",
        logs=str(tmp_path / "logs"),
        label="test",
        jitter=0.02,
        num_envs=2048,
        min_transitions=1_000_000_000,
    )


def test_billion_budget_is_rounded_up_and_resume_keeps_configuration(tmp_path):
    p = plan(tmp_path)
    updates = math.ceil(p["min_transitions"] / (p["num_envs"] * 24))
    assert updates == 20346 and updates * p["num_envs"] * 24 == 1_000_046_592
    for task in ("transfer", "push"):
        for condition in ("wrist", "wrist_static", "active"):
            run = dict(
                task=task,
                condition=condition,
                id=task + condition,
                seed=2,
                fixed_view=4 if task == "transfer" and condition == "wrist_static" else 0,
            )
            fresh = study.command_for(p, run, updates)
            resumed = study.command_for(p, run, updates, tmp_path / "model_1000.pt")
            assert resumed[:-2] == fresh
            assert resumed[-2] == "--resume"
            from active_perception_arms.train import experiment_from_args, parser

            cfg = experiment_from_args(parser().parse_args(fresh[3:]))
            assert cfg.manip_control == "cartesian" and cfg.reset_xy_jitter == 0.02


def test_submission_is_reserved_once_and_continuations_depend_on_parent(tmp_path, monkeypatch):
    p = plan(tmp_path)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(p))
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEPMGR", "must-be-removed")

    def submit(command, **kwargs):
        assert "SLURM_STEPMGR" not in kwargs["env"]
        calls.append(command)
        return SimpleNamespace(
            returncode=0, stdout=f"{100 + len(calls)}\n", stderr="", check_returncode=lambda: None
        )

    monkeypatch.setattr(study.subprocess, "run", submit)
    first = study.submit(path, "transfer-wrist_static-s0")
    second = study.submit(path, "transfer-wrist_static-s0")
    assert first == second and len(calls) == 1
    assert "--chdir=" + str(tmp_path) in calls[0]
    study.submit(path, "transfer-wrist_static-s0", continuation=True)
    assert "--dependency=afterany:123" in calls[1]
    for _ in range(6):
        study.submit(path, "transfer-wrist_static-s0", continuation=True)
    with pytest.raises(RuntimeError, match="Eight allocations"):
        study.submit(path, "transfer-wrist_static-s0", continuation=True)


@pytest.mark.parametrize("stop_signal", ["SIGUSR1", "SIGTERM"])
def test_worker_checkpoint_continuation_and_full_budget_without_real_training(
    tmp_path, monkeypatch, stop_signal
):
    import signal

    import torch

    p = plan(tmp_path)
    run = dict(
        task="push", condition="wrist_static", id="push-wrist_static-s0", seed=0, fixed_view=0
    )
    p.update(runs=[run], competence_gate=0.8)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(p))
    run_dir = Path(p["logs"]) / run["id"] / "unique-run"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment.json").write_text("{}")
    handlers, launches = {}, []
    interrupt = [True]
    monkeypatch.setattr(study, "verify_source", lambda _: None)
    monkeypatch.setattr(study, "build_report", lambda _: None)
    monkeypatch.setattr(study.signal, "signal", lambda sig, fn: handlers.update({sig: fn}))
    monkeypatch.setattr(study, "submit", lambda *a, **kw: launches.append(kw) or {"job_id": "next"})

    class Process:
        def __init__(self, command, **kwargs):
            if "active_perception_arms.train" in command:
                updates = int(command[command.index("--iterations") + 1])
                if interrupt[0]:
                    interrupt[0] = False
                    torch.save({"iter": 6}, run_dir / "interrupted.pt")
                    sig = getattr(signal, stop_signal)
                    handlers[sig](sig, None)
                else:
                    assert "--resume" in command
                    torch.save({"iter": updates - 1}, run_dir / f"model_{updates - 1}.pt")
            elif "active_perception_arms.evaluate" in command:
                episodes = int(command[command.index("--episodes") + 1])
                result = dict(
                    episodes=episodes,
                    successes=episodes,
                    success_rate=1.0,
                    outcomes=[True] * episodes,
                    termination_audit=dict(success_hold_violations=0, success_push_violations=0),
                )
                Path(command[command.index("--output") + 1]).write_text(json.dumps(result))

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(study.subprocess, "Popen", Process)
    study.work(path, run["id"])
    state_path = Path(p["output"]) / "runs" / run["id"] / "state.json"
    state = json.loads(state_path.read_text())
    assert state["status"] == (
        "checkpointed_for_continuation" if stop_signal == "SIGUSR1" else "stopped"
    )
    assert state["updates"] == 7 and state["transitions"] == 7 * 2048 * 24
    assert launches == ([{"continuation": True}] if stop_signal == "SIGUSR1" else [])
    study.work(path, run["id"])
    state = json.loads(state_path.read_text())
    assert state["status"] == "completed" and state["transitions"] == 1_000_046_592
    assert len(state["allocations"]) == 2
    assert len([r for r in state["evaluations"] if r["split"] == "test"]) == 2
    assert all(r["split"] == "validation" for r in state["competence_gate"]["evaluations"])


def test_suspension_blocks_new_submissions_and_already_queued_workers(tmp_path):
    p = plan(tmp_path)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(p))
    output = Path(p["output"])
    output.mkdir()
    (output / "suspension.json").write_text('{"automatic_expansion_allowed": false}')
    for continuation in (False, True):
        with pytest.raises(RuntimeError, match="submission is disabled"):
            study.submit(path, "transfer-wrist_static-s0", continuation=continuation)
    with pytest.raises(RuntimeError, match="worker is disabled"):
        study.work(path, "transfer-wrist_static-s0")
