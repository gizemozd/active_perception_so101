"""Physical action semantics, backend agreement, and bounded-policy density."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from active_perception_arms.bounded_distribution import SquashedGaussian
from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.environment import make_env_cfg
from active_perception_arms.native import NativeEnv


def config(task, **kw):
    return Experiment(
        task=task,
        control_revision="pose_v1",
        condition="active",
        num_envs=4,
        randomize=False,
        render_sensors=False,
        **kw,
    )


@pytest.mark.parametrize("task,revision", [("plug", None), ("push", "v1"), ("push", "push_v2")])
def test_shared_deltas_pause_bounds_and_native_warp_agreement(task, revision):
    from mjlab.envs import ManagerBasedRlEnv

    cfg = config(task, task_revision=revision)
    if revision == "push_v2":
        cfg = replace(
            cfg,
            push_target_layout="arc_v2",
            push_workspace_revision="arc_clear_v2",
            push_visibility_revision="crossbar_v3",
            reward_profile="push_dense",
            reset_xy_jitter=0.02,
        )
    native = NativeEnv(cfg)
    setup = make_env_cfg(cfg)
    setup.terminations = {}
    env = ManagerBasedRlEnv(setup, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["proprio"].shape == (4, 51)
        term = env.action_manager.get_term("arms")
        before_tcp = term.tcp_target.clone()
        before_cam = term.gimbal_target.clone()
        assert cfg.action_dim == 8
        action = torch.tensor([[0.2, -0.1, 0.3, 0.1, -0.2, 0.1, 0.2, -0.1]]).repeat(4, 1)
        env.step(action)
        native.step(action[0].numpy())
        torch.testing.assert_close(term.tcp_target, before_tcp)  # inspection pause
        torch.testing.assert_close(
            term.gimbal_target, before_cam + action[:, 3:] * term.camera_pose_ik.delta
        )
        np.testing.assert_allclose(native.gimbal_target, term.gimbal_target[0].numpy(), atol=1e-6)
        env.episode_length_buf[:] = 25
        native.step_count = 25
        before_tcp = term.tcp_target.clone()
        env.step(action)
        native.step(action[0].numpy())
        torch.testing.assert_close(term.tcp_target, before_tcp + 0.005 * action[:, :3])
        np.testing.assert_allclose(native.tcp_target, term.tcp_target[0].numpy(), atol=1e-5)
        np.testing.assert_allclose(
            native.targets["camera_arm"][:5], term.targets["camera_arm"][0, :5].numpy(), atol=2e-4
        )
        # Saturated inputs cannot escape pose workspace or leave stale state after subset reset.
        for _ in range(30):
            term.process_actions(torch.full((4, 8), 100.0))
        assert torch.all(term.gimbal_target <= term.camera_pose_ik.high + 1e-6)
        saved = term.gimbal_target[1:].clone()
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term.gimbal_target[0], term.camera_pose_ik.home)
        torch.testing.assert_close(term.gimbal_target[1:], saved)
    finally:
        env.close()


@pytest.mark.parametrize("task", ["plug", "push"])
def test_camera_decimation_and_fixed_home(task):
    cfg = config(task, camera_interval=4)
    env = NativeEnv(cfg)
    action = np.ones(8) * 0.1
    env.step(action)
    pose = env.gimbal_target.copy()
    targets = env.targets["camera_arm"].copy()
    for _ in range(3):
        env.step(action)
    np.testing.assert_array_equal(env.gimbal_target, pose)
    np.testing.assert_array_equal(env.targets["camera_arm"], targets)
    env.step(action)
    assert not np.array_equal(env.gimbal_target, pose)
    static = NativeEnv(replace(cfg, condition="wrist_static"))
    home = static.targets["camera_arm"].copy()
    for _ in range(30):
        static.step(np.zeros(3))
    np.testing.assert_array_equal(home, static.targets["camera_arm"])


def test_old_configs_keep_old_mapping_and_new_roundtrip():
    legacy = saved_experiment({"task": "plug", "task_revision": "hidden_prongs_v1"})
    assert legacy.control_revision == "legacy_v1" and legacy.plug_control == "absolute"
    for task in ("plug", "push"):
        cfg = config(task)
        assert saved_experiment(cfg.to_dict()) == cfg


@pytest.mark.parametrize("task", ["plug", "push"])
@pytest.mark.parametrize("condition", ["wrist", "static", "wrist_static", "initial", "active"])
def test_fresh_training_defaults_to_shared_deltas(task, condition):
    from active_perception_arms.train import experiment_from_args, parser

    cfg = experiment_from_args(parser().parse_args(["--task", task, "--condition", condition]))
    assert cfg.control_revision == "pose_v1"
    assert cfg.translation_control == "delta"
    assert (cfg.width, cfg.height, cfg.manip_dim) == (128, 96, 3)


@pytest.mark.parametrize("task", ["plug", "push"])
def test_calibrated_inspection_start_is_shared_and_tracks_zero_delta(task):
    cfg = config(task, camera_workspace="inspection_v2", camera_start="inspection_v1")
    active = NativeEnv(cfg)
    static = NativeEnv(replace(cfg, condition="wrist_static"))
    np.testing.assert_array_equal(active.data.qpos, static.data.qpos)
    camera = active.model.camera("camera_arm/wrist_cam").id
    requested = active.camera_pose_ik.initial[:3].numpy()
    assert np.linalg.norm(active.data.cam_xpos[camera] - requested) < 0.005
    for _ in range(25):
        active.step(np.zeros(8))
    assert np.linalg.norm(active.data.cam_xpos[camera] - requested) < 0.008
    np.testing.assert_allclose(active.gimbal_target, active.camera_pose_ik.initial.numpy())


def test_squashed_density_matches_transformed_distribution_and_gradients():
    torch.manual_seed(9)
    dist = SquashedGaussian(8, init_std=0.4, std_range=(0.05, 1.0))
    raw = torch.randn(32, 8, requires_grad=True)
    dist.update(raw)
    action = dist.sample()
    assert torch.all(action.abs() < 1)
    reference = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(2 * torch.tanh(raw / 2), torch.full_like(raw, 0.4)),
        [torch.distributions.TanhTransform()],
    )
    torch.testing.assert_close(dist.log_prob(action), reference.log_prob(action).sum(-1))
    loss = -dist.log_prob(action).mean() - 0.01 * dist.entropy.mean()
    loss.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    assert torch.isfinite(dist.std_param.grad).all()
    assert dist.deterministic_output(torch.full((1, 8), 1000.0)).max() < 1
