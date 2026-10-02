import numpy as np
import pytest

from active_perception_arms.scenes import look_at_quat
from active_perception_arms.screening_diagnostics import Renderer, kinematic_path
from active_perception_arms.task_screening import (
    CANDIDATES,
    ScreenScene,
    collect,
    fixed_catalog,
    knowledge,
    summarize,
)


def observations(fixed, active=None, wrist=None):
    if active is None:
        active = fixed.copy()
    if wrist is None:
        wrist = np.zeros(fixed.shape[:2], dtype=bool)
    return {
        "fixed": fixed,
        "active": active,
        "wrist": wrist,
        "active_wrist": np.zeros_like(active),
        "fixture_intrusions": [],
        "max_tcp_error_m": 0.0,
    }


def test_memory_cannot_recover_an_unobserved_new_state():
    seen = np.zeros((1, 3, 1), dtype=bool)
    seen[:, 0] = True
    wrist = np.zeros((1, 3), dtype=bool)
    assert knowledge(seen, wrist, fresh=False).all()
    assert not knowledge(seen, wrist, fresh=True).any()


def test_view_selection_uses_validation_even_when_test_prefers_another_view():
    val = np.zeros((2, 3, 2), dtype=bool)
    val[:, :, 0] = True
    test = ~val
    result = summarize(CANDIDATES[2], observations(val), observations(test))
    assert result["best_static_index"] == 0
    assert result["best_static_plus_wrist"] == 0
    assert result["oracle_initial_fixed_plus_wrist"] == 1


def test_two_view_scan_is_stronger_than_one_scheduled_view_per_phase():
    seen = np.zeros((2, 3, 2), dtype=bool)
    seen[0, 1, 0] = seen[0, 2, 1] = True
    seen[1, 1, 1] = seen[1, 2, 0] = True
    result = summarize(CANDIDATES[3], observations(seen), observations(seen))
    assert result["oracle_initial_reachable_plus_wrist"] == 0
    assert result["scheduled_pair_plus_wrist"] == 0.5
    assert result["two_view_scan_plus_wrist"] == 1
    assert result["oracle_moving_plus_wrist"] == 1


def test_moving_arm_does_not_inherit_wrist_visibility_from_a_different_arm_pose():
    hidden = np.zeros((2, 3, 2), dtype=bool)
    parked_wrist = np.ones((2, 3), dtype=bool)
    data = observations(hidden, wrist=parked_wrist)
    result = summarize(CANDIDATES[3], data, data)
    assert result["best_static_plus_wrist"] == 1
    assert result["oracle_moving_plus_wrist"] == 0


def test_fixed_search_includes_overhead_and_all_active_endpoints():
    reachable = [{"position": [-0.18, -0.22, 0.16], "matrix": np.eye(3).tolist()}]
    fixed = fixed_catalog(reachable)
    assert len(fixed) > 350
    assert fixed[-1] == reachable[0]
    assert any(np.allclose(v["position"], [0, 0.025, 0.6]) for v in fixed)


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_query_snapshots_have_no_fixture_penetration_and_reachable_hand_pose(candidate):
    scene = ScreenScene(candidate)
    result = collect(scene, [20000], [], [])
    assert not result["fixture_intrusions"]
    assert result["max_tcp_error_m"] < 0.003


def test_fresh_offset_changes_without_leaking_into_hand_proprioception():
    scene = ScreenScene(CANDIDATES[2])
    scene.set_episode(20000, 1)
    hand = scene.data.qpos[scene.qadr["manipulator"]].copy()
    feature = scene.feature_center.copy()
    scene.set_episode(20000, 2)
    np.testing.assert_allclose(scene.data.qpos[scene.qadr["manipulator"]], hand)
    assert not np.allclose(scene.feature_center, feature)
    assert scene.epoch == 2
    scene.set_episode(20000, 1)
    np.testing.assert_allclose(scene.feature_center, feature)


def test_two_site_geometry_does_not_teleport_between_queries():
    scene = ScreenScene(CANDIDATES[4])
    scene.set_episode(20000, 1)
    housings = scene.data.mocap_pos[[scene.housing_id, scene.second_housing_id]].copy()
    source = scene.feature_center.copy()
    scene.set_episode(20000, 2)
    np.testing.assert_allclose(
        scene.data.mocap_pos[[scene.housing_id, scene.second_housing_id]], housings
    )
    assert np.linalg.norm(source - scene.feature_center) > 0.1


def test_camera_route_rejects_joint_limit_violation():
    scene = ScreenScene(CANDIDATES[0])
    start = scene.home["camera_arm"][:5].copy()
    end = start.copy()
    end[0] = 100
    result = kinematic_path(scene, start, end)
    assert not result["feasible"] and result["reason"] == "joint_limit"


@pytest.mark.render
@pytest.mark.parametrize("phase", [0, 1])
def test_pixel_center_ray_counts_match_actual_segmentation(phase):
    import mujoco

    scene = ScreenScene(CANDIDATES[3])
    scene.set_episode(20000, phase)
    position = np.array([-0.10, -0.30, 0.24])
    quat = look_at_quat(position, scene.feature_center)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    view = {"position": position.tolist(), "matrix": matrix.reshape(3, 3).tolist()}
    predicted = scene.visibility(position, matrix.reshape(3, 3))["marker_pixels"]
    renderer = Renderer(scene, (96, 72))
    try:
        seg = renderer.frame("screen_fixed", view, segmentation=True)
    finally:
        renderer.close()
    actual = (
        (seg[:, :, 0] == scene.feature_geom) & (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
    ).sum()
    assert actual > 0  # Prevent a vacuous comparison of two fully occluded views.
    assert abs(int(actual) - predicted) <= 1
