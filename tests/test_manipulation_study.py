"""Protect Cartesian camera routing, physical scoring and historical schemas."""

import mujoco
import numpy as np
import pytest
import torch

from active_perception_arms import mdp, pushing
from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.environment import make_env
from active_perception_arms.native import NativeEnv, ScriptedPolicy


def config(task, **kwargs):
    return Experiment(
        task=task,
        task_revision="pick_place_v2" if task == "transfer" else "push_v2",
        manip_control="cartesian",
        reward_profile=task + "_dense",
        **kwargs,
    )


@pytest.mark.parametrize("task,dim", [("transfer", 4), ("push", 3)])
def test_cartesian_camera_routing_and_partial_reset(task, dim):
    torch.set_num_threads(1)
    cfg = config(task, num_envs=2, render_sensors=False, initial_seconds=0)
    env = make_env(cfg, "cpu")
    try:
        env.reset()
        term = env.action_manager.get_term("arms")
        assert cfg.action_dim == dim + 5
        start = term.tcp_target.clone()
        camera = term.targets["camera_arm"].clone()
        action = torch.zeros(2, cfg.action_dim)
        action[:, dim:] = 0.25
        term.process_actions(action)
        assert (term.tcp_target - start).abs().max() <= term.cartesian.delta + 1e-7
        torch.testing.assert_close(
            term.targets["camera_arm"][:, :5], camera[:, :5] + cfg.joint_step * 0.25
        )
        before = mdp.proprio(env).clone()
        s = mdp.state(env)
        s.transfer_lifted[:] = True
        env.sim.data.qpos[:, s.object_qpos_address] += 0.02
        torch.testing.assert_close(mdp.proprio(env), before)
        retained = term.tcp_target[1].clone()
        mdp.reset_task(env, torch.tensor([0]), cfg)
        torch.testing.assert_close(term.tcp_target[0], s.home["tcp"])
        torch.testing.assert_close(term.tcp_target[1], retained)
        obs, reward, *_ = env.step(action)
        assert torch.isfinite(reward).all() and obs["critic"].dtype == torch.float32
    finally:
        env.close()


@pytest.mark.parametrize("layout", ["square_v1", "arc_v1"])
def test_push_contact_at_goal_is_not_a_success(layout):
    torch.set_num_threads(1)
    env = NativeEnv(config("push", condition="wrist_static", randomize=False,
                           render_sensors=False, push_target_layout=layout))
    policy = ScriptedPolicy(env)
    rejected_contact = False
    hold = 0
    for _ in range(300):
        success = env.step(policy())
        close = np.linalg.norm(env.object_pos[:2] - env.goal[:2]) < env.cfg.push_success_tolerance
        tool_close = np.linalg.norm(env.data.geom("pushing_tip").xpos - env.object_pos) <= 0.025
        if close and (not pushing.native_clear(env) or tool_close):
            assert not success
            rejected_contact = True
        hold = hold + 1 if success else 0
        if hold >= 3:
            break
    assert rejected_contact and hold >= 3
    assert policy.stage == 2


def test_historical_configs_and_invalid_new_profiles():
    assert saved_experiment({"task": "push"}).manip_dim == 5
    assert saved_experiment({"task": "transfer"}).manip_dim == 6
    with pytest.raises(ValueError):
        Experiment(task="plug", manip_control="cartesian")
    with pytest.raises(ValueError):
        Experiment(task="push", reward_profile="push_dense")
    with pytest.raises(ValueError):
        config("push", success_state_sample="derived_substep")


def test_current_push_qpos_and_tcp_are_used_before_forward():
    torch.set_num_threads(1)
    cfg = config("push", num_envs=2, render_sensors=False)
    env = make_env(cfg, "cpu")
    proxy = NativeEnv(cfg)
    try:
        env.reset()
        s = mdp.state(env)
        env.sim.data.qpos[:, s.manip_qpos_addresses[0]] += 0.1
        proxy.data.qpos[proxy.qadr["manipulator"][:5]] = env.sim.data.qpos[
            0, s.manip_qpos_addresses
        ]
        mujoco.mj_forward(proxy.model, proxy.data)
        tcp, obj, *_ = mdp.positions(env)
        tool = pushing.tool_position(env, s)
        np.testing.assert_allclose(
            tool[0],
            proxy.data.geom("pushing_tip").xpos + env.scene.env_origins[0].numpy(),
            atol=2e-6,
        )
        np.testing.assert_allclose(tcp[0], proxy.tcp + env.scene.env_origins[0].numpy(), atol=2e-6)
        torch.testing.assert_close(
            obj, env.sim.data.qpos[:, s.object_qpos_address : s.object_qpos_address + 3]
        )
    finally:
        env.close()
