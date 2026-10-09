import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from active_perception_arms import mdp
from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.train import initial_std_from_args, main, parser


def reward_env(monkeypatch, profile="legacy_log_hold"):
    cfg = Experiment(reward_profile=profile)
    state = SimpleNamespace(
        cfg=cfg,
        object_qpos_address=0,
        hold=torch.zeros(1, dtype=torch.long),
        succeeded=torch.zeros(1, dtype=torch.bool),
        last_success_step=-1,
        potential_valid=torch.zeros(1, dtype=torch.bool),
        previous_potential=torch.zeros(1),
    )
    # Deliberately stale and distant derived position: shaping must use current
    # freejoint distance and success must still require three physical samples.
    derived = torch.tensor([[0.05, 0.0, 0.0]])
    qpos = torch.tensor([[0.0015, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    env = SimpleNamespace(
        task_state=state,
        common_step_counter=0,
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
        scene={
            "object": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=torch.zeros(1, 3)))
        },
        action_manager=SimpleNamespace(action=torch.zeros(1, cfg.action_dim)),
    )
    monkeypatch.setattr(
        mdp, "positions", lambda _: (None, derived, None, torch.zeros_like(derived))
    )
    monkeypatch.setattr(mdp, "potential", lambda _: torch.ones(1))
    return env


def test_log_reward_requires_hold_and_uses_current_physical_distance(monkeypatch):
    env = reward_env(monkeypatch)
    shaping_and_time = -math.log1p(100 * 0.0015) - 0.005
    for step in (1, 2):
        env.common_step_counter = step
        assert mdp.task_reward(env).item() == pytest.approx(shaping_and_time, abs=1e-6)
    env.common_step_counter = 3
    assert mdp.task_reward(env).item() == pytest.approx(1000 + shaping_and_time, abs=1e-4)
    assert env.task_state.hold.item() == 3
    env.sim.data.qpos[:, 0] = 0.003
    env.common_step_counter = 4
    assert mdp.task_reward(env).item() < 0 and env.task_state.hold.item() == 0


def test_log_reward_prefers_nearer_unsolved_states(monkeypatch):
    env = reward_env(monkeypatch)
    env.sim.data.qpos[:, 0] = 0.03
    far = mdp.task_reward(env).item()
    env.sim.data.qpos[:, 0] = 0.004
    env.common_step_counter = 1
    near = mdp.task_reward(env).item()
    assert far < near < 0
    assert not env.task_state.succeeded.item()


@pytest.mark.parametrize("profile", ("progress", "legacy_log_hold"))
def test_profiles_keep_identical_action_penalty(monkeypatch, profile):
    env = reward_env(monkeypatch, profile)
    env.sim.data.qpos[:, 0] = 0.02
    before = mdp.task_reward(env).item()
    env.action_manager.action[:] = 1
    after = mdp.task_reward(env).item()
    expected = 0.0005 * env.task_state.cfg.action_dim / env.task_state.cfg.manip_dim
    assert before - after == pytest.approx(expected, abs=1e-6)


def test_reward_profile_scope_and_old_config_default():
    with pytest.raises(ValueError, match="legacy_log_hold requires"):
        Experiment(task="transfer", reward_profile="legacy_log_hold")
    with pytest.raises(ValueError, match="legacy_log_hold requires"):
        Experiment(reward_profile="legacy_log_hold", success_state_sample="derived_substep")
    old = Experiment().to_dict()
    old.pop("reward_profile")
    assert saved_experiment(old).reward_profile == "progress"


def test_resume_inherits_profile_and_std_rejecting_changes(tmp_path, capsys):
    cfg = Experiment(reward_profile="legacy_log_hold")
    (tmp_path / "experiment.json").write_text(json.dumps(cfg.to_dict()))
    (tmp_path / "runner.json").write_text(
        json.dumps(
            {
                "algorithm": {
                    "learning_rate": 0.0003,
                    "schedule": "adaptive",
                    "entropy_coef": 0.003,
                },
                "actor": {"distribution_cfg": {"init_std": 1.0}},
            }
        )
    )
    args = ["--task", "plug", "--resume", str(tmp_path / "model_99.pt"), "--dry-run"]
    main(args)
    report = json.loads(capsys.readouterr().out)
    assert (
        report["experiment"]["reward_profile"] == "legacy_log_hold" and report["initial_std"] == 1.0
    )
    assert not report["training_started"]
    with pytest.raises(ValueError, match="Resume reward_profile differs"):
        main([*args, "--reward-profile", "progress"])
    with pytest.raises(ValueError, match="Resume initial_std differs"):
        main([*args, "--initial-std", "0.4"])
    main([*args, "--initial-std", "1.0", "--reward-profile", "legacy_log_hold"])
    assert not json.loads(capsys.readouterr().out)["training_started"]


@pytest.mark.parametrize("value", ("0", "-1", "nan", "inf"))
def test_initial_std_must_be_positive_finite(value):
    with pytest.raises(ValueError, match="positive and finite"):
        initial_std_from_args(parser().parse_args(["--task", "plug", "--initial-std", value]))


def test_fresh_initial_std_default_and_explicit_value(capsys):
    main(["--task", "plug", "--dry-run"])
    assert json.loads(capsys.readouterr().out)["initial_std"] == 0.4
    main(["--task", "plug", "--initial-std", "1.0", "--dry-run"])
    assert json.loads(capsys.readouterr().out)["initial_std"] == 1.0


def test_slurm_environment_passes_reward_and_initial_std():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "scripts/slurm/train_plug.sbatch"],
        cwd=root,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "SLURM_ARRAY_TASK_ID": "6",
            "PROJECT_ROOT": str(root),
            "REWARD_PROFILE": "legacy_log_hold",
            "INITIAL_STD": "1.0",
            "SUCCESS_STATE_SAMPLE": "current_qpos",
            "RESUME": "",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout[result.stdout.index("{") :])
    assert report["initial_std"] == 1.0
    assert report["experiment"]["reward_profile"] == "legacy_log_hold"
    assert report["experiment"]["success_state_sample"] == "current_qpos"
