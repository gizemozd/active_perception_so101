import json
from types import SimpleNamespace

import pytest
import torch

from active_perception_arms import mdp
from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.train import main


@pytest.mark.parametrize("sample,expected", [("derived_substep", True), ("current_qpos", False)])
def test_success_hold_uses_selected_physical_sample(monkeypatch, sample, expected):
    cfg = Experiment(success_state_sample=sample)
    s = SimpleNamespace(
        cfg=cfg,
        object_qpos_address=0,
        hold=torch.zeros(1, dtype=torch.long),
        succeeded=torch.zeros(1, dtype=torch.bool),
        last_success_step=-1,
    )
    # Reproduce the measured failure: derived body position is inside tolerance
    # while the integrated freejoint has already moved outside tolerance.
    derived = torch.tensor([[0.0015, 0.0, 0.0]])
    qpos = torch.tensor([[0.0025, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    env = SimpleNamespace(
        task_state=s,
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
        scene={
            "object": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=torch.zeros(1, 3)))
        },
        common_step_counter=0,
    )
    monkeypatch.setattr(
        mdp, "positions", lambda _: (None, derived, None, torch.zeros_like(derived))
    )
    for step in range(1, 4):
        env.common_step_counter = step
        assert bool(mdp.success(env).item()) == (expected and step == 3)
        before = s.hold.clone()
        mdp.success(env)  # Reward/metric rechecks must not advance the streak.
        assert torch.equal(s.hold, before)
    if sample == "current_qpos":
        qpos[:, 0] = 0.001
        derived[:, 0] = 0.003  # Opposite lag: current physical state is valid.
        for step in range(4, 7):
            env.common_step_counter = step
            assert bool(mdp.success(env).item()) == (step == 6)
        qpos[:, 0] = 0.0025
        env.common_step_counter = 7
        assert not mdp.success(env).item() and s.hold.item() == 0


def test_new_default_and_old_saved_config_sampling_are_explicit():
    cfg = Experiment()
    assert cfg.success_state_sample == "current_qpos"
    old = cfg.to_dict()
    old.pop("success_state_sample")
    assert saved_experiment(old).success_state_sample == "derived_substep"
    assert "success_state_sample" not in old  # Deserialization cannot mutate source.
    assert saved_experiment(cfg.to_dict()).success_state_sample == "current_qpos"
    with pytest.raises(ValueError, match="success_state_sample"):
        Experiment(success_state_sample="post_reset")


def test_training_resume_inherits_old_sampling_and_rejects_override(tmp_path, capsys):
    cfg = Experiment(num_envs=256)
    old = cfg.to_dict()
    old.pop("success_state_sample")
    (tmp_path / "experiment.json").write_text(json.dumps(old))
    (tmp_path / "runner.json").write_text(
        json.dumps(
            {
                "algorithm": {"learning_rate": 3e-4, "schedule": "adaptive", "entropy_coef": 0.003},
                "actor": {"distribution_cfg": {"init_std": 0.4}},
            }
        )
    )
    args = ["--task", "plug", "--resume", str(tmp_path / "model_99.pt"), "--dry-run"]
    main(args)
    result = json.loads(capsys.readouterr().out)
    assert result["experiment"]["success_state_sample"] == "derived_substep"
    assert not result["training_started"]
    with pytest.raises(ValueError, match="success_state_sample"):
        main([*args, "--success-state-sample", "current_qpos"])
    main(["--task", "plug", "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["experiment"]["success_state_sample"] == "current_qpos"
