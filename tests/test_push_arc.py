"""Object-relative sampling, precise scoring, and historical checkpoint compatibility."""

import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from active_perception_arms import mdp, pushing
from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.environment import make_env
from active_perception_arms.evaluation_audit import TerminationAudit
from active_perception_arms.native import NativeEnv
from active_perception_arms.train import experiment_from_args, parser


def config(**kwargs):
    return Experiment(
        task="push",
        task_revision="push_v2",
        manip_control="cartesian",
        reward_profile="push_dense",
        condition="wrist",
        occlusion="clean",
        render_sensors=False,
        reset_xy_jitter=0.02,
        **kwargs,
    )


def test_arc_sampling_endpoints_and_numpy_torch_agree():
    samples = np.random.default_rng(31).random((10000, 2))
    samples[:3] = [[0, 0], [0.5, 0.5], [1, 1]]
    offsets = pushing.arc_offset(samples)
    np.testing.assert_allclose(offsets, pushing.arc_offset(torch.tensor(samples)).numpy())
    angles = np.rad2deg(np.arctan2(offsets[:, 0], offsets[:, 1]))
    radii = np.linalg.norm(offsets, axis=1)
    np.testing.assert_allclose(angles[:3], [-80, 0, 80])
    np.testing.assert_allclose(radii[:3], [0.05, 0.085, 0.12])
    assert angles.min() >= -80 and angles.max() <= 80
    assert radii.min() >= 0.05 - 1e-12 and radii.max() <= 0.12 + 1e-12
    old_radii = np.linalg.norm(pushing.arc_offset(samples, "arc_v1"), axis=1)
    np.testing.assert_allclose(old_radii[:3], [0.05, 0.06, 0.07])


def test_new_default_and_saved_checkpoint_resume(tmp_path):
    cfg = config()
    assert cfg.push_target_layout == "arc_v2" and cfg.push_success_tolerance == 0.005
    old = cfg.to_dict()
    old.pop("push_target_layout")
    loaded = saved_experiment(old)
    assert loaded.push_target_layout == "square_v1"
    assert loaded.push_success_tolerance == 0.015 and loaded.push_marker_half_size == 0.026
    saved_arc_v1 = saved_experiment({**old, "push_target_layout": "arc_v1"})
    assert saved_arc_v1.push_target_layout == "arc_v1"
    np.testing.assert_allclose(
        pushing.arc_offset(np.array([1.0, 1.0]), saved_arc_v1.push_target_layout),
        [0.07 * np.sin(np.deg2rad(80)), 0.07 * np.cos(np.deg2rad(80))],
    )
    (tmp_path / "experiment.json").write_text(json.dumps(old))
    command = ["--task", "push", "--resume", str(tmp_path / "model.pt")]
    assert experiment_from_args(parser().parse_args(command)).push_target_layout == "square_v1"
    with pytest.raises(ValueError, match="push_target_layout"):
        experiment_from_args(parser().parse_args(command + ["--push-target-layout", "arc_v2"]))
    assert saved_experiment(cfg.to_dict()) == cfg


def test_native_arc_resets_marker_and_precision():
    torch.set_num_threads(1)
    env = NativeEnv(config())
    angles = []
    for seed in range(64):
        env.reset(seed)
        offset = env.goal[:2] - env.object_pos[:2]
        assert 0.05 - 1e-8 <= np.linalg.norm(offset) <= 0.12 + 1e-8
        angles.append(np.rad2deg(np.arctan2(*offset)))
    assert min(angles) < -65 and max(angles) > 65
    np.testing.assert_allclose(env.model.geom("fixture/marker").size[:2], [0.01, 0.01])
    env.reset(40)
    for distance, expected in [(0.0049, True), (0.0051, False), (0.012, False)]:
        env.data.qpos[env.object_adr : env.object_adr + 3] = env.goal + [distance, 0, 0]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        assert env.success() == expected
    env.reset(40)
    first = env.goal.copy()
    env.reset(40)
    np.testing.assert_array_equal(env.goal, first)
    fixed = NativeEnv(replace(env.cfg, randomize=False))
    np.testing.assert_allclose(fixed.goal[:2] - fixed.object_pos[:2], [0, 0.085])


def test_warp_cpu_arc_subset_reset_and_scoring_audit():
    torch.set_num_threads(1)
    env = make_env(config(num_envs=4), "cpu")
    audit = TerminationAudit(env)
    try:
        env.reset()
        s = mdp.state(env)
        _, obj, _, goal = mdp.positions(env)
        delta = goal[:, :2] - obj[:, :2]
        radius = delta.norm(dim=-1)
        assert ((radius >= 0.05 - 1e-6) & (radius <= 0.12 + 1e-6)).all()
        assert (torch.atan2(delta[:, 0], delta[:, 1]).abs() <= pushing.ARC_HALF_ANGLE).all()
        before = env.sim.data.mocap_pos[1:].clone()
        mdp.reset_task(env, torch.tensor([0]), s.cfg)
        torch.testing.assert_close(env.sim.data.mocap_pos[1:], before)
        _, _, _, goal = mdp.positions(env)
        pos = env.sim.data.qpos[:, s.object_qpos_address : s.object_qpos_address + 3]
        pos[:] = goal
        pos[:, 0] += torch.tensor([0.004, 0.006, 0.012, 0.020])
        env.sim.data.qvel[:] = 0
        expected = torch.tensor([True, False, False, False])
        torch.testing.assert_close(mdp.instantaneous_success(env), expected)
        audit.reset_batch()
        for _ in range(3):
            env.common_step_counter += 1
            env.termination_manager.compute()
        torch.testing.assert_close(audit.snapshot["push_criterion_history"][:, -1], expected)
        assert audit.snapshot["success"][0]
    finally:
        audit.close()
        env.close()


def test_wide_controls_are_opt_in_and_saved_configs_preserve_them(tmp_path):
    from active_perception_arms.manipulation import bounds

    original = config()
    assert original.push_control_revision == "legacy_v1"
    np.testing.assert_allclose(bounds("push")[1], [0.065, 0.11, 0.13])
    wide = replace(original, push_control_revision="wide_v2")
    assert saved_experiment(wide.to_dict()) == wide
    lo, hi = np.array(bounds("push", "wide_v2"))
    # All requested arc endpoints and rear approach poses fit the new action box.
    for dx in [-0.02, 0.02]:
        for dy in [-0.02, 0.02]:
            for theta in np.linspace(-80, 80, 17):
                direction = np.array([np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))])
                source = np.array([-0.025 + dx, -0.045 + dy])
                for xy in [source - 0.035 * direction, source + (0.12 - 0.017) * direction]:
                    assert ((xy >= lo[:2]) & (xy <= hi[:2])).all()
    (tmp_path / "experiment.json").write_text(json.dumps(wide.to_dict()))
    args = ["--task", "push", "--resume", str(tmp_path / "model.pt")]
    assert experiment_from_args(parser().parse_args(args)).push_control_revision == "wide_v2"
    with pytest.raises(ValueError, match="push_control_revision"):
        experiment_from_args(parser().parse_args(args + ["--push-control-revision", "legacy_v1"]))


def test_clear_arc_workspace_removes_only_legacy_partition():
    from active_perception_arms.scenes import native_spec

    old = config(push_visibility_revision="crossbar_v3")
    revised = replace(old, push_workspace_revision="arc_clear_v2")
    a, b = native_spec(old).compile(), native_spec(revised).compile()
    assert a.geom("table/partition").contype > 0
    assert mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_GEOM, "table/partition") == -1
    assert b.geom("visibility/crossbar").contype == 0
    assert a.nq == b.nq and a.ngeom == b.ngeom + 1
    assert old.push_success_tolerance == revised.push_success_tolerance == 0.005
    assert old.push_target_layout == revised.push_target_layout == "arc_v2"
    historical = old.to_dict()
    historical.pop("push_workspace_revision")
    historical.pop("push_control_revision")
    assert saved_experiment(historical) == old
