import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from active_perception_arms import mdp, plug_learning
from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.train import experiment_from_args, initial_std_from_args, main, parser


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


def test_insertion_reward_boundary_and_distance_values():
    obj = torch.zeros(6, 3, dtype=torch.float64)
    obj[:, 0] = torch.tensor([0, 0.001999, 0.002, 0.003, 0.01, 0.03], dtype=obj.dtype)
    reward = plug_learning.insertion_reward(obj, torch.zeros_like(obj))
    torch.testing.assert_close(
        reward,
        torch.tensor(
            [1000, 1000, -math.log(1.2), -math.log(1.3), -math.log(2), -math.log(4)],
            dtype=obj.dtype,
        ),
    )


def test_insertion_bonus_is_exact_instantaneous_and_does_not_advance_hold(monkeypatch):
    env = reward_env(monkeypatch, "plug_insertion")
    # Current qpos is within tolerance, while the mocked derived pose is 50 mm away.
    env.action_manager.action[:] = 1
    assert mdp.task_reward(env).item() == 1000
    assert env.task_state.hold.item() == 0
    for step in (1, 2, 3):
        env.common_step_counter = step
        assert mdp.success(env).item() == (step == 3)
        assert mdp.task_reward(env).item() == 1000
        assert env.task_state.hold.item() == step


@pytest.mark.parametrize("condition", ("wrist", "wrist_static", "active"))
def test_insertion_has_no_action_time_or_stage_cost(monkeypatch, condition):
    env = reward_env(monkeypatch, "plug_insertion")
    env.task_state.cfg = Experiment(condition=condition)
    env.sim.data.qpos[:, 0] = 0.03
    env.action_manager.action = torch.ones(1, env.task_state.cfg.action_dim)
    env.task_state.plug_raised = env.task_state.plug_aligned = torch.tensor([True])
    expected = -math.log(4)
    assert mdp.task_reward(env).item() == pytest.approx(expected)
    env.action_manager.action.zero_()
    env.task_state.plug_raised = env.task_state.plug_aligned = torch.tensor([False])
    assert mdp.task_reward(env).item() == pytest.approx(expected)


def test_insertion_optional_rotation_matches_upstream_quaternion_metric():
    obj = torch.tensor([[0.03, 0.0, 0.0], [0.001, 0.0, 0.0]], dtype=torch.float64)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=obj.dtype)
    reward = plug_learning.insertion_reward(obj, torch.zeros_like(obj), quat)
    assert reward[0].item() == pytest.approx(-math.log(1 + 100 * math.sqrt(0.03**2 + 2)))
    assert reward[1].item() == 1000  # Upstream bonus tests position, even with rotation.


def test_insertion_default_scope_and_resume_compatibility(tmp_path, capsys):
    assert Experiment().reward_profile == "plug_insertion"
    assert Experiment(plug_control="delta5mm").reward_profile == "plug_insertion"
    assert Experiment(task="push").reward_profile == "progress"
    assert Experiment(task="transfer").reward_profile == "transfer_staged"
    assert Experiment(success_state_sample="derived_substep").reward_profile == "progress"
    for options in ({"task": "push"}, {"success_state_sample": "derived_substep"}):
        with pytest.raises(ValueError, match="plug_insertion requires"):
            Experiment(reward_profile="plug_insertion", **options)
    main(["--task", "plug", "--dry-run"])
    assert json.loads(capsys.readouterr().out)["experiment"]["reward_profile"] == "plug_insertion"
    for profile in ("progress", "legacy_log_hold", "plug_staged", "plug_insertion"):
        cfg = Experiment(reward_profile=profile, plug_control="delta5mm")
        (tmp_path / "experiment.json").write_text(json.dumps(cfg.to_dict()))
        args = ["--task", "plug", "--resume", str(tmp_path / "model.pt"), "--dry-run"]
        assert experiment_from_args(parser().parse_args(args)).reward_profile == profile
        if profile != "plug_insertion":
            with pytest.raises(ValueError, match="Resume reward_profile differs"):
                experiment_from_args(
                    parser().parse_args([*args, "--reward-profile", "plug_insertion"])
                )


def test_insertion_reward_real_environment_uses_variant_goal_and_current_qpos():
    from mjlab.envs import ManagerBasedRlEnv

    from active_perception_arms.environment import make_env_cfg

    cfg = Experiment(num_envs=4, render_sensors=False)
    env_cfg = make_env_cfg(cfg)
    assert env_cfg.scale_rewards_by_dt is False
    assert env_cfg.rewards["task"].weight == 1
    env = ManagerBasedRlEnv(env_cfg, device="cpu")
    try:
        env.reset(seed=9)
        s = mdp.state(env)
        goal = mdp.positions(env)[3].clone()
        socket = env.scene["fixture"].data.site_pos_w[:, 0]
        torch.testing.assert_close(goal[:, :2], socket[:, :2] - s.plug_offsets)
        # Do not forward derived state: both actor-independent critic state and
        # reward must observe these newly integrated physical coordinates.
        env.sim.data.qpos[:, s.object_qpos_address : s.object_qpos_address + 3] = goal
        env.sim.data.qpos[2:, s.object_qpos_address] += 0.03
        rewards = mdp.task_reward(env)
        torch.testing.assert_close(rewards, torch.tensor([1000, 1000, -math.log(4), -math.log(4)]))
        assert s.hold.tolist() == [0, 0, 0, 0]
    finally:
        env.close()


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
