"""Budget, matched camera settings, and interruption continuation without training."""

import importlib.util
import json
import signal
from pathlib import Path

import torch

from active_perception_arms.train import experiment_from_args, parser

SPEC = importlib.util.spec_from_file_location(
    "plug_precision", Path(__file__).resolve().parents[1] / "scripts/run_plug_precision.py"
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def plan(tmp_path):
    return dict(
        root=str(tmp_path),
        source=str(tmp_path),
        output=str(tmp_path / "out"),
        logs=str(tmp_path / "logs"),
        python="python",
        num_envs=512,
        updates=1465,
        transitions_per_policy=18_001_920,
        milestones=[750, 1465],
    )


def test_budget_and_fresh_resume_commands_match_across_camera_conditions(tmp_path):
    p = plan(tmp_path)
    assert p["updates"] * p["num_envs"] * 24 == p["transitions_per_policy"]
    assert p["transitions_per_policy"] >= 18_000_000
    for condition in ("wrist", "wrist_static", "active"):
        case = dict(condition=condition, fixed_view=7 if condition == "wrist_static" else 0)
        fresh = study.command_for(p, case, 1465)
        resumed = study.command_for(p, case, 1465, Path("model.pt"))
        assert resumed[:-2] == fresh
        cfg = experiment_from_args(parser().parse_args(fresh[3:]))
        assert cfg.plug_control == "delta5mm" and cfg.reward_profile == "plug_staged"
        assert cfg.manip_dim == 3 and cfg.actor_observation == "rgb"
        assert cfg.episode_seconds == 3.5 and cfg.success_state_sample == "current_qpos"


def test_interrupted_worker_preserves_updates_and_stops_at_authorized_budget(tmp_path, monkeypatch):
    p = plan(tmp_path)
    case = dict(condition="wrist_static", fixed_view=7)
    p["cases"] = [case]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(p))
    (tmp_path / "out").mkdir()
    run = tmp_path / "logs/wrist_static/only-run"
    run.mkdir(parents=True)
    (run / "experiment.json").write_text("{}")
    handlers, submissions, commands = {}, [], []
    interrupt = [True]
    monkeypatch.setattr(study, "verify", lambda _: None)
    monkeypatch.setattr(study, "report", lambda _: None)
    monkeypatch.setattr(study.signal, "signal", lambda sig, fn: handlers.update({sig: fn}))
    monkeypatch.setattr(
        study, "submit", lambda *a, **kw: submissions.append(kw) or {"job_id": "next"}
    )

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            if "active_perception_arms.train" in command:
                updates = int(command[command.index("--iterations") + 1])
                if interrupt[0]:
                    interrupt[0] = False
                    torch.save({"iter": 6}, run / "interrupted.pt")
                    handlers[signal.SIGUSR1](signal.SIGUSR1, None)
                else:
                    assert "--resume" in command
                    torch.save({"iter": updates - 1}, run / f"model_{updates - 1}.pt")
            elif "active_perception_arms.evaluate" in command:
                n = int(command[command.index("--episodes") + 1])
                result = dict(
                    episodes=n,
                    successes=0,
                    outcomes=[False] * n,
                    per_variant={
                        v: dict(episodes=n // 4, successes=0) for v in ("xm", "xp", "ym", "yp")
                    },
                    termination_audit=dict(
                        success_hold_violations=0,
                        success_three_sample_violations=0,
                        success_qpos_above_distance_threshold=0,
                    ),
                )
                Path(command[command.index("--output") + 1]).write_text(json.dumps(result))

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(study.subprocess, "Popen", Process)
    study.work(path, "wrist_static")
    state_path = tmp_path / "out/wrist_static/state.json"
    state = json.loads(state_path.read_text())
    assert state["updates"] == 7 and state["transitions"] == 7 * 512 * 24
    assert state["status"] == "checkpointed_for_continuation"
    assert submissions == [dict(continuation=True)]
    study.work(path, "wrist_static")
    state = json.loads(state_path.read_text())
    assert state["status"] == "completed"
    assert state["transitions"] == 18_001_920 and state["updates"] == 1465
    assert len(state["evaluations"]) == 4
    assert len([r for r in state["evaluations"] if r["split"] == "test"]) == 2
    assert all(
        int(c[c.index("--iterations") + 1]) <= 1465
        for c in commands
        if "active_perception_arms.train" in c
    )
