"""Precision control, physical stage history, actor isolation and saved semantics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from active_perception_arms import mdp, plug, plug_learning
from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.native import NativeEnv
from active_perception_arms.train import experiment_from_args, parser


def test_stages_require_physical_clearance_and_alignment():
    tcp = torch.tensor([[0.0, 0.0, 0.080], [0.0, 0.0, 0.09], [0.0, 0.0, 0.09]])
    obj = torch.tensor([[0.0, 0.0, 0.016], [0.02, 0.0, 0.045], [0.0, 0.0, 0.045]])
    goal = torch.tensor([[0.0, 0.0, 0.016]]).repeat(3, 1)
    raised, aligned = torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)
    plug_learning.update_stages(tcp, obj, goal, raised, aligned)
    assert raised.tolist() == [False, True, True]
    assert aligned.tolist() == [False, False, True]
    reward = plug_learning.density(tcp, obj, goal, raised, aligned)
    assert reward[0] == 0  # fake goal-only state has no lift history
    assert reward[2] > reward[1] > reward[0]
    obj[2, 2] = 0.016
    inserted = plug_learning.density(tcp, obj, goal, raised, aligned)[2]
    assert inserted == 3
    obj[2, 0] = 0.06  # history cannot retain precision credit after drifting away
    assert plug_learning.density(tcp, obj, goal, raised, aligned)[2] < 1.02


def test_new_options_are_explicit_and_resume_rejects_control_change(tmp_path):
    old = Experiment().to_dict()
    old.pop("plug_control")
    assert saved_experiment(old).plug_control == "absolute"
    with pytest.raises(ValueError, match="plug_staged"):
        Experiment(reward_profile="plug_staged")
    with pytest.raises(ValueError, match="plug task"):
        Experiment(task="transfer", plug_control="delta5mm")
    cfg = Experiment(plug_control="delta5mm", reward_profile="plug_staged")
    (tmp_path / "experiment.json").write_text(json.dumps(cfg.to_dict()))
    args = parser().parse_args(["--task", "plug", "--resume", str(tmp_path / "model.pt")])
    restored = experiment_from_args(args)
    assert restored.plug_control == "delta5mm" and restored.reward_profile == "plug_staged"
    args.plug_control = "absolute"
    with pytest.raises(ValueError, match="Resume plug_control"):
        experiment_from_args(args)


def test_incremental_native_control_respects_pause_and_small_commands():
    env = NativeEnv(Experiment(condition="wrist", plug_control="delta5mm", render_sensors=False))
    before = env.tcp_target.copy()
    action = np.array([0.2, -0.2, 0.2])
    env.step(action)
    np.testing.assert_array_equal(env.tcp_target, before)
    env.step_count = 25
    env.step(action)
    np.testing.assert_allclose(env.tcp_target - before, [0.001, -0.001, 0.001], atol=1e-9)
    for _ in range(30):
        env.step(np.ones(3))
    assert np.all(env.tcp_target <= plug.ACTION_HIGH)


def test_warp_controls_raw_physics_stage_reset_and_actor_isolation():
    from mjlab.envs import ManagerBasedRlEnv

    from active_perception_arms.environment import make_env_cfg

    torch.set_num_threads(1)
    cfg = Experiment(
        num_envs=4,
        condition="active",
        plug_control="delta5mm",
        reward_profile="plug_staged",
        render_sensors=False,
    )
    env = ManagerBasedRlEnv(make_env_cfg(cfg), device="cpu")
    try:
        env.reset(seed=9)
        term, s = env.action_manager.get_term("arms"), mdp.state(env)
        env.episode_length_buf[:] = 25
        before = term.tcp_target.clone()
        camera = term.gimbal_target.clone()
        action = torch.zeros(4, cfg.action_dim)
        action[:, :3] = torch.tensor([0.2, -0.2, 0.2])
        term.process_actions(action)
        torch.testing.assert_close(
            term.tcp_target - before, torch.tensor([[0.001, -0.001, 0.001]]).repeat(4, 1)
        )
        torch.testing.assert_close(term.gimbal_target, camera)
        tcp, obj, _, _ = mdp.positions(env)
        # FK reads actual qpos, not the modified command or stale derived sites.
        assert not torch.allclose(tcp, term.tcp_target)
        env.sim.data.qpos[0, s.object_qpos_address] += 0.02
        new_obj = mdp.positions(env)[1]
        assert new_obj[0, 0] == env.sim.data.qpos[0, s.object_qpos_address]
        proprio = mdp.proprio(env).clone()
        critic = mdp.critic_state(env).clone()
        s.plug_raised[:] = s.plug_aligned[:] = True
        torch.testing.assert_close(mdp.proprio(env), proprio)
        assert not torch.equal(mdp.critic_state(env), critic)
        mdp.reset_task(env, torch.tensor([1]), cfg)
        assert s.plug_raised.tolist() == [True, False, True, True]
        assert s.plug_aligned.tolist() == [True, False, True, True]
    finally:
        env.close()


def test_camera_dimensions_do_not_change_manipulator_effort(monkeypatch):
    def reward(camera_dims):
        cfg = Experiment(plug_control="delta5mm", reward_profile="plug_staged")
        state = SimpleNamespace(cfg=cfg)
        env = SimpleNamespace(
            action_manager=SimpleNamespace(
                action=torch.tensor([[0.2, 0.3, 0.4] + [1.0] * camera_dims])
            )
        )
        monkeypatch.setattr(mdp, "state", lambda _: state)
        monkeypatch.setattr(mdp, "positions", lambda _: (None, None, None, None))
        monkeypatch.setattr(plug_learning, "features", lambda *_: torch.tensor([1.0]))
        monkeypatch.setattr(mdp, "success", lambda _: torch.tensor([False]))
        return mdp.task_reward(env)

    torch.testing.assert_close(reward(0), reward(5))
