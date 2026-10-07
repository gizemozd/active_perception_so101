import numpy as np
import pytest

from active_perception_arms.config import CONDITIONS, TASKS, Experiment, static_candidates
from active_perception_arms.native import NativeEnv, ScriptedPolicy, occluder_position
from active_perception_arms.scenes import native_spec


@pytest.mark.parametrize("task", TASKS)
def test_scripted_task_is_physically_solvable(task):
    env = NativeEnv(Experiment(task=task, num_envs=1, randomize=False, occlusion="clean"))
    assert not env.success()
    script = ScriptedPolicy(env)
    held = 0
    for _ in range(200):
        held = held + 1 if env.step(script()) else 0
        assert np.isfinite(env.data.qpos).all()
    assert held >= 3
    assert np.linalg.norm(env.object_pos[:2] - env.goal[:2]) < 0.02
    env.reset(0)
    assert not env.success()


@pytest.mark.parametrize("condition", CONDITIONS)
def test_camera_modes_and_shared_initial_manipulator_hold(condition):
    exp = Experiment(condition=condition, num_envs=1, randomize=False, occlusion="clean")
    env = NativeEnv(exp)
    start = {name: q.copy() for name, q in env.targets.items()}
    action = np.full(exp.action_dim, 0.2)
    env.step(action)
    np.testing.assert_allclose(env.targets["manipulator"], start["manipulator"])
    if condition in ("active", "initial"):
        assert not np.allclose(env.targets["camera_arm"], start["camera_arm"])
    else:
        np.testing.assert_allclose(env.targets["camera_arm"], start["camera_arm"])
    env.step_count = int(exp.initial_seconds / exp.step_dt) + 5
    camera = env.targets["camera_arm"].copy()
    env.step(action)
    assert not np.allclose(env.targets["manipulator"], start["manipulator"])
    if condition in ("active", "scheduled"):
        assert not np.allclose(camera, env.targets["camera_arm"])
    else:
        np.testing.assert_allclose(camera, env.targets["camera_arm"])


def test_occlusion_is_optical_and_seeded():
    cfg = Experiment(num_envs=1, occlusion="random")
    env = NativeEnv(cfg)
    q = env.data.qpos.copy()
    onset = env.onset
    env.reset(0)
    np.testing.assert_array_equal(q, env.data.qpos)
    assert onset == env.onset
    panel = env.model.geom("occluder/panel")
    assert panel.contype == 0 and panel.conaffinity == 0
    assert occluder_position(3.5, 3.0, 2.0, 1.0, cfg)[2] > 0
    assert occluder_position(1.0, 3.0, 2.0, 1.0, cfg)[2] < 0
    env.reset(1)
    assert not np.array_equal(q, env.data.qpos) or env.onset != onset


def test_static_grid_has_real_overhead_and_equal_intrinsics():
    assert len(static_candidates()) == 26
    assert (0.0, 0.025, 0.6) in static_candidates()
    model = native_spec(Experiment()).compile()
    sensing = [
        model.camera(name).id for name in ("manipulator/wrist_cam", "camera_arm/wrist_cam", "fixed")
    ]
    np.testing.assert_allclose(model.cam_fovy[sensing], model.cam_fovy[sensing[0]])
    assert model.ncam == 5
    assert model.nu == 12


@pytest.mark.render
def test_wrist_view_contains_task_object_and_metrics():
    from active_perception_arms.sanity import CameraAudit

    env = NativeEnv(Experiment(task="transfer", randomize=False, occlusion="clean", num_envs=1))
    audit = CameraAudit(env)
    try:
        image, metrics = audit.render("manipulator/wrist_cam")
        assert image.shape == (72, 96, 3)
        assert metrics["object_pixels"] > 20
        assert metrics["table_fraction"] < 0.95
    finally:
        audit.close()
