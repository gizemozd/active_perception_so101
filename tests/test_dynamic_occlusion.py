from dataclasses import replace

import numpy as np
import pytest
import torch

from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.native import NativeEnv, ScriptedPolicy, occluder_position
from active_perception_arms.occlusion import (
    dynamic_position_numpy,
    dynamic_position_torch,
    sample_dynamic_numpy,
    sample_dynamic_torch,
    with_occlusion,
)


@pytest.mark.parametrize("task", ("plug", "transfer", "push"))
def test_dynamic_defaults_fit_horizon_and_old_checkpoint_defaults_are_unchanged(task):
    legacy = Experiment(task=task, success_state_sample="derived_substep")
    assert legacy.occlusion == ("clean" if task == "plug" else "random")
    old = {
        k: v
        for k, v in legacy.to_dict().items()
        if not k.startswith("dynamic_") and k not in ("occlusion_revision", "success_state_sample")
    }
    assert saved_experiment(old) == legacy
    dynamic = replace(legacy, occlusion="dynamic")
    assert dynamic.episode_seconds == legacy.episode_seconds
    assert dynamic.dynamic_onset_range[1] + dynamic.dynamic_duration_range[1] < (
        dynamic.episode_seconds - dynamic.step_dt
    )
    assert dynamic.occlusion_revision == "world_sweep_v1"
    # JSON round trips preserve dynamic tuples. Existing fixed camera fields
    # intentionally retain the legacy list/tuple behavior.
    import json

    restored = saved_experiment(json.loads(json.dumps(dynamic.to_dict())))
    for name, value in dynamic.to_dict().items():
        if name.startswith("dynamic_"):
            assert getattr(restored, name) == value
    assert json.dumps(restored.to_dict()) == json.dumps(dynamic.to_dict())


@pytest.mark.parametrize(
    "params",
    [
        {"dynamic_onset_range": (2.5, 2.0)},
        {"dynamic_duration_range": (0, 1)},
        {"dynamic_duration_range": (2, 3)},
        {"dynamic_center": (0, np.nan, 0.1)},
        {"dynamic_travel_range": (-0.1, 0.1)},
        {"dynamic_panel_half_size": (0.04, 0, 0.07)},
        {"dynamic_panel_yaw": np.inf},
        {"dynamic_center_jitter": (0, -0.01, 0)},
    ],
)
def test_invalid_dynamic_parameters(params):
    with pytest.raises(ValueError):
        Experiment(occlusion="dynamic", **params)


def test_dynamic_knobs_require_explicit_mode():
    with pytest.raises(ValueError, match="occlusion=dynamic"):
        Experiment(dynamic_travel_range=(0.1, 0.2))


def test_evaluation_mode_switch_clears_only_inactive_dynamic_parameters():
    cfg = Experiment(occlusion="dynamic")
    unchanged = with_occlusion(cfg, None, num_envs=8)
    assert unchanged.dynamic_onset_range == cfg.dynamic_onset_range
    assert unchanged.num_envs == 8
    clean = with_occlusion(cfg, "clean")
    assert clean == Experiment(occlusion="clean")
    restored = with_occlusion(clean, "dynamic")
    assert restored == cfg


def test_numpy_torch_trajectory_parity_including_onset_end_and_direction():
    times = np.array([0.0, 0.5, 0.75, 1.0, 1.5, 1.50001, 2.0])
    center = np.array([0.02, 0.10, 0.13])
    for side in (-1.0, 1.0):
        expected = np.stack(
            [dynamic_position_numpy(t, 0.5, 1.0, side, center, 0.12) for t in times]
        )
        actual = dynamic_position_torch(
            torch.tensor(times),
            torch.full((7,), 0.5),
            torch.ones(7),
            torch.full((7,), side),
            torch.tensor(center).repeat(7, 1),
            torch.full((7,), 0.12, dtype=torch.float64),
        )
        np.testing.assert_allclose(actual, expected)
        assert np.all(expected[[0, 5, 6], 2] == -1)
        assert expected[1, 0] == pytest.approx(center[0] - side * 0.06)
        assert expected[4, 0] == pytest.approx(center[0] + side * 0.06)


def test_parameter_sampling_bounds_both_directions_and_seed_reproducibility():
    cfg = Experiment(occlusion="dynamic")
    a = sample_dynamic_numpy(np.random.default_rng(17), cfg)
    b = sample_dynamic_numpy(np.random.default_rng(17), cfg)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    torch.manual_seed(17)
    params = sample_dynamic_torch(4096, cfg, "cpu")
    for key, bounds in [
        ("onset", cfg.dynamic_onset_range),
        ("duration", cfg.dynamic_duration_range),
        ("travel", cfg.dynamic_travel_range),
    ]:
        assert params[key].min() >= bounds[0] and params[key].max() <= bounds[1]
    assert set(params["side"].tolist()) == {-1.0, 1.0}
    center_error = (params["center"] - torch.tensor(cfg.dynamic_center)).abs()
    assert (center_error <= torch.tensor(cfg.dynamic_center_jitter) + 1e-7).all()
    assert (params["onset"] + params["duration"] < cfg.episode_seconds).all()


def test_native_policy_independent_occluder_world_distribution_and_seeded_reset():
    cfg = Experiment(occlusion="dynamic", seed=13, num_envs=1)
    reference = NativeEnv(cfg)
    for condition in ("wrist", "wrist_static", "initial", "scheduled", "active"):
        env = NativeEnv(replace(cfg, condition=condition))
        assert (env.onset, env.duration, env.side, env.panel_travel) == (
            reference.onset,
            reference.duration,
            reference.side,
            reference.panel_travel,
        )
        np.testing.assert_array_equal(env.panel_center, reference.panel_center)
        np.testing.assert_array_equal(env.data.qpos, reference.data.qpos)
    reference.reset(13)
    np.testing.assert_array_equal(reference.data.qpos, env.data.qpos)
    assert reference.model.geom("occluder/panel").contype == 0
    assert reference.model.geom("occluder/panel").conaffinity == 0


def test_optical_sweep_preserves_physics_and_allows_waiting_for_recovery():
    cfg = Experiment(occlusion="dynamic", randomize=False, num_envs=1)
    env, clean = (
        NativeEnv(cfg),
        NativeEnv(
            replace(
                cfg,
                occlusion="clean",
                occlusion_revision=None,
                dynamic_onset_range=None,
                dynamic_duration_range=None,
                dynamic_center=None,
                dynamic_center_jitter=None,
                dynamic_travel_range=None,
                dynamic_panel_half_size=None,
                dynamic_panel_yaw=None,
            )
        ),
    )
    policy = ScriptedPolicy(clean)
    for _ in range(int(np.ceil(cfg.episode_seconds / cfg.step_dt))):
        action = policy()
        env.step(action)
        clean.step(action)
        np.testing.assert_array_equal(env.data.qpos, clean.data.qpos)
    assert clean.success() and env.success()
    after = occluder_position(
        env.onset + env.duration + cfg.step_dt,
        env.onset,
        env.duration,
        env.side,
        cfg,
        center=env.panel_center,
        travel=env.panel_travel,
    )
    assert after[2] < 0


def test_dynamic_cli_arguments(capsys):
    import json

    from active_perception_arms.train import main

    main(
        [
            "--task",
            "plug",
            "--occlusion",
            "dynamic",
            "--dynamic-onset-range",
            "0.2",
            "0.8",
            "--dynamic-duration-range",
            "0.3",
            "0.9",
            "--dynamic-center",
            "0",
            "0.1",
            "0.12",
            "--dynamic-center-jitter",
            "0.01",
            "0.01",
            "0.01",
            "--dynamic-travel-range",
            "0.12",
            "0.16",
            "--dynamic-panel-yaw",
            "0.2",
            "--dynamic-panel-half-size",
            "0.04",
            "0.004",
            "0.06",
            "--dry-run",
        ]
    )
    cfg = json.loads(capsys.readouterr().out)["experiment"]
    assert cfg["occlusion"] == "dynamic"
    assert cfg["dynamic_duration_range"] == [0.3, 0.9]
    assert cfg["dynamic_panel_yaw"] == 0.2


@pytest.fixture(scope="module")
def dynamic_env():
    from active_perception_arms.environment import make_env

    cfg = Experiment(
        occlusion="dynamic",
        num_envs=4,
        render_sensors=False,
        randomize=False,
        dynamic_panel_yaw=0.3,
    )
    env = make_env(cfg, "cpu")
    env.reset()
    yield env
    env.close()


def test_warp_subset_reset_preserves_other_panels_and_privileged_parameters(dynamic_env):
    from active_perception_arms import mdp

    env = dynamic_env
    s = mdp.state(env)
    times = s.onset + s.duration * 0.5
    mdp.update_panel(env, times)
    panel = env.sim.mj_model.body("occluder/panel").mocapid[0]
    poses = env.sim.data.mocap_pos[:, panel].clone()
    cached = s.panel_pose.clone()
    params = torch.cat(
        (
            s.panel_center,
            s.panel_travel[:, None],
            s.onset[:, None],
            s.duration[:, None],
            s.side[:, None],
        ),
        dim=-1,
    ).clone()
    env.reset(env_ids=torch.tensor([1, 3]))
    torch.testing.assert_close(env.sim.data.mocap_pos[[0, 2], panel], poses[[0, 2]])
    torch.testing.assert_close(s.panel_pose[[0, 2]], cached[[0, 2]])
    after = torch.cat(
        (
            s.panel_center,
            s.panel_travel[:, None],
            s.onset[:, None],
            s.duration[:, None],
            s.side[:, None],
        ),
        dim=-1,
    )
    torch.testing.assert_close(after[[0, 2]], params[[0, 2]])
    assert not torch.equal(after[[1, 3]], params[[1, 3]])
    assert (env.sim.data.mocap_pos[[1, 3], panel, 2] < 0).all()
    for i in (0, 2):
        expected = dynamic_position_numpy(
            float(times[i]),
            float(s.onset[i]),
            float(s.duration[i]),
            float(s.side[i]),
            s.panel_center[i].numpy(),
            float(s.panel_travel[i]),
        )
        np.testing.assert_allclose(poses[i] - env.scene.env_origins[i], expected, atol=1e-7)


def test_new_occluder_privileged_parameters_are_critic_only(dynamic_env):
    from active_perception_arms import mdp

    env = dynamic_env
    actor = mdp.proprio(env).clone()
    critic = mdp.critic_state(env).clone()
    s = mdp.state(env)
    s.panel_center += 0.005
    s.panel_travel += 0.02
    s.onset += 0.05
    s.duration += 0.05
    s.side *= -1
    torch.testing.assert_close(actor, mdp.proprio(env))
    assert not torch.equal(critic, mdp.critic_state(env))
    assert critic.shape == (4, 80) and actor.shape == (4, 51)


@pytest.mark.render
def test_actual_rendered_obstruction_and_same_state_clearance_recovery():
    import mujoco

    from active_perception_arms.config import static_candidates
    from active_perception_arms.dynamic_occlusion_diagnostics import paired_views
    from active_perception_arms.sanity import CameraAudit

    cfg = Experiment(
        occlusion="dynamic",
        num_envs=1,
        randomize=False,
        fixed_position=static_candidates("plug")[7],
        dynamic_center_jitter=(0, 0, 0),
        dynamic_travel_range=(0.12, 0.12),
    )
    env = NativeEnv(cfg)
    env.data.mocap_pos[env.panel_id] = env.panel_center
    mujoco.mj_forward(env.model, env.data)
    pose = env.data.mocap_pos[env.panel_id].copy()
    audit = CameraAudit(env)
    try:
        metrics, current, clear = paired_views(env, audit)
        np.testing.assert_array_equal(env.data.mocap_pos[env.panel_id], pose)
        assert metrics["scripted moving external"]["object_pixels_removed"] > 30
        assert metrics["fixed"]["useful_pixels_removed"] > 0
        assert all(row["object_pixels_removed"] >= 0 for row in metrics.values())
        assert not np.array_equal(current[2], clear[2])
        env.data.mocap_pos[env.panel_id, 2] = -1
        mujoco.mj_forward(env.model, env.data)
        for camera, expected in zip(
            ("manipulator/wrist_cam", "fixed", "camera_arm/wrist_cam"), clear, strict=True
        ):
            actual, _ = audit.render(camera)
            np.testing.assert_array_equal(actual, expected)
    finally:
        audit.close()


@pytest.mark.gpu
@pytest.mark.parametrize("task", ("plug", "transfer", "push"))
def test_dynamic_cuda_graph_steps_and_partial_reset(task):
    from active_perception_arms import mdp
    from active_perception_arms.environment import make_env

    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA GPU unavailable on this host")
    cfg = Experiment(task=task, occlusion="dynamic", num_envs=8, width=32, height=32)
    env = make_env(cfg, "cuda:0")
    try:
        obs, _ = env.reset()
        for _ in range(5):
            obs, reward, _, _, _ = env.step(torch.zeros(8, cfg.action_dim, device="cuda"))
        torch.cuda.synchronize()
        assert obs["external"].is_cuda and torch.isfinite(reward).all()
        s = mdp.state(env)
        # Render during the event, not only its hidden pre-onset state, and
        # compare actual device mocap positions against the native trajectory.
        env.episode_length_buf[:] = ((s.onset + s.duration * 0.5) / cfg.step_dt).long()
        control_time = env.episode_length_buf.clone() * cfg.step_dt
        obs, reward, terminated, truncated, _ = env.step(
            torch.zeros(8, cfg.action_dim, device="cuda")
        )
        assert not (terminated | truncated).any()
        panel = env.sim.mj_model.body("occluder/panel").mocapid[0]
        actual = (env.sim.data.mocap_pos[:, panel] - env.scene.env_origins).cpu().numpy()
        expected = np.stack(
            [
                dynamic_position_numpy(
                    float(control_time[i]),
                    float(s.onset[i]),
                    float(s.duration[i]),
                    float(s.side[i]),
                    s.panel_center[i].cpu().numpy(),
                    float(s.panel_travel[i]),
                )
                for i in range(8)
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        assert (actual[:, 2] > 0).all()
        assert obs["external"].max() > obs["external"].min()
        other_pose = s.panel_pose[2:].clone()
        env.reset(env_ids=torch.tensor([0, 1], device="cuda"))
        torch.testing.assert_close(s.panel_pose[2:], other_pose)
        assert torch.isfinite(obs["critic"]).all()
    finally:
        env.close()
