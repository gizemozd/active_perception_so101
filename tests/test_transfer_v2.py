"""Transfer must physically pick up, support, release and withdraw the cube."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from active_perception_arms import mdp, transfer
from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.native import NativeEnv


def feature_case(
    *,
    lifted=True,
    fixed=False,
    moving=False,
    touching=False,
    floor=True,
    speed=0.0,
    angular=0.0,
    hand_distance=0.06,
    position_error=0.0,
):
    obj = torch.tensor([[position_error, 0.0, 0.058]])
    tcp = obj + torch.tensor([hand_distance, 0.0, 0.0])
    return transfer.features(
        tcp,
        obj,
        torch.tensor([[0.0, 0.0, 0.058]]),
        torch.tensor([speed]),
        torch.tensor([angular]),
        {
            k: torch.tensor([v])
            for k, v in dict(fixed=fixed, moving=moving, manipulator=touching, floor=floor).items()
        },
        torch.tensor([lifted]),
    )


@pytest.mark.parametrize(
    "override",
    [
        dict(lifted=False),
        dict(touching=True),
        dict(floor=False),
        dict(speed=0.1),
        dict(angular=1.0),
        dict(hand_distance=0.02),
        dict(position_error=0.025),
    ],
)
def test_goal_proximity_alone_cannot_complete_transfer(override):
    assert feature_case()["instantaneous_success"].item()
    assert not feature_case(**override)["instantaneous_success"].item()


def test_staged_reward_encourages_release_and_withdrawal(monkeypatch):
    held = feature_case(fixed=True, moving=True, touching=True, hand_distance=0.008)
    released = feature_case(hand_distance=0.015)
    withdrawn = feature_case()
    assert transfer.potential(held) < transfer.potential(released) < transfer.potential(withdrawn)
    cfg = Experiment(task="transfer")
    state = SimpleNamespace(
        cfg=cfg, previous_potential=torch.tensor([4.0]), potential_valid=torch.tensor([True])
    )
    env = SimpleNamespace(
        task_state=state,
        action_manager=SimpleNamespace(action=torch.zeros(1, cfg.action_dim)),
        termination_manager=SimpleNamespace(dones=torch.tensor([False])),
    )
    monkeypatch.setattr(mdp, "potential", lambda _: torch.tensor([4.0]))
    monkeypatch.setattr(mdp, "success", lambda _: torch.tensor([False]))
    assert mdp.task_reward(env).item() < 0  # holding a high-potential state pays no positive reward
    state.previous_potential[:] = 4.0
    env.termination_manager.dones[:] = True
    timeout = mdp.task_reward(env).item()
    assert timeout < -12.0
    state.previous_potential[:] = 4.0
    monkeypatch.setattr(mdp, "success", lambda _: torch.tensor([True]))
    assert mdp.task_reward(env).item() == pytest.approx(timeout + transfer.SUCCESS_BONUS)


def test_a_placed_cube_without_pickup_history_is_rejected_in_real_native_physics():
    env = NativeEnv(Experiment(task="transfer", randomize=False, render_sensors=False))
    assert env.model.geom("object/block").condim == 6
    legacy = NativeEnv(Experiment(task="transfer", task_revision="v1", render_sensors=False))
    assert legacy.model.geom("object/block").condim == 3
    env.data.qpos[env.object_adr : env.object_adr + 3] = env.goal
    mujoco.mj_forward(env.model, env.data)
    assert np.linalg.norm(env.object_pos - env.goal) < 1e-10
    assert not env.success()
    assert not env.transfer_lifted


def test_revision_and_reward_defaults_preserve_old_saved_transfer():
    fresh = Experiment(task="transfer")
    assert fresh.task_revision == transfer.REVISION and fresh.reward_profile == "transfer_staged"
    saved = saved_experiment({"task": "transfer"})
    assert saved.task_revision == "v1" and saved.reward_profile == "progress"
    with pytest.raises(ValueError, match="current_qpos"):
        Experiment(task="transfer", success_state_sample="derived_substep")
    with pytest.raises(ValueError, match="transfer_staged"):
        Experiment(task="plug", reward_profile="transfer_staged")


def test_current_qpos_fk_contact_sensors_and_subset_reset():
    from mjlab.envs import ManagerBasedRlEnv

    from active_perception_arms.environment import make_env_cfg

    torch.set_num_threads(1)
    cfg = Experiment(task="transfer", num_envs=4, render_sensors=False, randomize=False)
    env = ManagerBasedRlEnv(make_env_cfg(cfg), device="cpu")
    proxy = NativeEnv(cfg)
    try:
        env.reset()
        s = mdp.state(env)
        assert mdp.critic_state(env).dtype == torch.float32
        before = mdp.proprio(env).clone()
        s.transfer_lifted[:] = True
        torch.testing.assert_close(mdp.proprio(env), before)  # no privileged actor leakage
        env.sim.data.qpos[:, s.manip_qpos_addresses[0]] += 0.08
        proxy.data.qpos[proxy.qadr["manipulator"][:5]] = env.sim.data.qpos[
            0, s.manip_qpos_addresses
        ].numpy()
        mujoco.mj_forward(proxy.model, proxy.data)
        tcp, obj, _, _ = mdp.positions(env)
        np.testing.assert_allclose(
            tcp[0].numpy(), proxy.tcp + env.scene.env_origins[0].numpy(), atol=2e-6
        )
        torch.testing.assert_close(
            obj, env.sim.data.qpos[:, s.object_qpos_address : s.object_qpos_address + 3]
        )
        assert not mdp.instantaneous_success(env).any()
        mdp.reset_task(env, torch.tensor([0]), cfg)
        assert not s.transfer_lifted[0] and s.transfer_lifted[1:].all()
        assert all(
            env.scene["transfer_" + name].data.found.shape[0] == 4
            for name in ("fixed", "moving", "manipulator", "floor")
        )
        _, reward, *_ = env.step(torch.zeros(4, cfg.action_dim))
        assert torch.isfinite(reward).all()
    finally:
        env.close()
