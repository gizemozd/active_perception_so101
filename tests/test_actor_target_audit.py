"""Diagnostic hooks must preserve real recurrent inference and physical snapshots."""

import copy
import importlib.util
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from active_perception_arms.config import Experiment
from active_perception_arms.policy import VisualMemoryModel, runner_cfg

spec = importlib.util.spec_from_file_location(
    "actor_target_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit_plug_actor_targets.py",
)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_passive_hooks_preserve_single_recurrent_forward_and_normalization():
    torch.manual_seed(17)
    obs = TensorDict(
        {
            "proprio": torch.randn(4, 43),
            "wrist": torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8),
            "external": torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8),
        },
        [4],
    )
    exp = Experiment(condition="wrist_static", num_envs=4, width=32, height=32)
    cfg = asdict(runner_cfg(exp))
    options = {k: v for k, v in cfg["actor"].items() if v is not None}
    options.pop("class_name")
    actor = VisualMemoryModel(obs, cfg["obs_groups"], "actor", 3, **options).eval()
    reference = copy.deepcopy(actor)
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    hooks = audit_module.PassiveFeatures(actor, exp.sensors)
    try:
        with torch.inference_mode():
            for step in range(3):
                hooks.begin(capture=step == 1)
                actual = actor(obs)
                features = hooks.end()
                expected = reference(obs)
                assert torch.equal(actual, expected)
                assert torch.equal(actor.get_hidden_state(), reference.get_hidden_state())
                if step == 1:
                    assert features["gru"].shape == (1, 4, 128)
                    assert features["cnn_wrist"].shape == (4, 64)
                    assert features["normalized_proprio"].shape == (4, 43)
                    assert torch.equal(features["actor"], actual)
                    # Hook storage is separate; mutating it cannot alter real state/output.
                    features["actor"].zero_()
                    assert torch.equal(actual, expected)
                else:
                    assert features == {}
            assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())
            hooks.begin()
            actor(obs)
            actor(obs)
            with pytest.raises(RuntimeError, match="Single-forward invariant"):
                hooks.end()
    finally:
        hooks.close()


def test_physical_snapshot_keeps_current_body_and_target_before_reset(monkeypatch):
    qpos = torch.tensor([[1.013, 2.020, 0.016, 1.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([[0.013, 0.0185, 0.061]])
    arms = SimpleNamespace(tcp_target=target, raw_action=torch.tensor([[0.2, 0.1, -0.5]]))
    env = SimpleNamespace(
        sim=SimpleNamespace(
            mj_model=SimpleNamespace(joint=lambda name: SimpleNamespace(qposadr=[0])),
            data=SimpleNamespace(qpos=qpos),
        ),
        action_manager=SimpleNamespace(get_term=lambda name: arms),
        episode_length_buf=torch.tensor([54]),
        device="cpu",
        num_envs=1,
    )
    task = SimpleNamespace(
        cfg=SimpleNamespace(task="plug", success_state_sample="current_qpos"),
        hold=torch.tensor([3]),
    )
    flags = {"success": torch.tensor([True]), "failure": torch.tensor([False])}
    env.termination_manager = SimpleNamespace(
        compute=lambda: torch.tensor([True]),
        get_term=lambda name: flags[name],
        time_outs=torch.tensor([False]),
    )
    goal = qpos[:, :3].clone()
    derived_body = goal + torch.tensor([[0.003, 0.0, 0.0]])
    monkeypatch.setattr(
        audit_module.mdp, "positions", lambda actual_env: (goal + 0.045, derived_body, goal, goal)
    )
    monkeypatch.setattr(audit_module.mdp, "state", lambda actual_env: task)
    audit = audit_module.PhysicalAudit(env)
    audit.reset_batch()
    try:
        audit.compute()
        assert audit.snapshot["qpos_position_error_m"].item() == 0
        assert audit.snapshot["derived_position_error_m"].item() > 0.002
        assert torch.equal(audit.snapshot["processed_tcp_target_local_m"], target)
        qpos.zero_()
        target.zero_()
        assert audit.snapshot["body_qpos_world_m"][0, 0].item() > 1
        assert audit.snapshot["processed_tcp_target_local_m"][0, 0].item() > 0.01
    finally:
        audit.close()


def test_fixed_ridge_uses_training_scale_and_intercept_only():
    train_x = np.array([[0.0, 7.0], [2.0, 7.0], [4.0, 7.0]])
    train_y = np.array([[1.0], [5.0], [9.0]])
    test_x = np.array([[1000.0, 7.0], [-1000.0, 7.0]])
    alpha = 1e-3
    # Constant second feature must have zero centered coefficient, not division by0.
    scale = train_x[:, 0].std()
    x = (train_x[:, 0] - 2.0) / scale
    slope = np.dot(x, train_y[:, 0] - 5.0) / (np.dot(x, x) + alpha)
    expected = ((test_x[:, 0] - 2.0) / scale * slope + 5.0)[:, None]
    assert np.allclose(audit_module.ridge_prediction(train_x, train_y, test_x), expected)


def test_time_bins_include_only_phase_specific_samples():
    trace = {
        "variant_id": np.array([0, 0, 1]),
        "action_time_s": np.array([0.96, 1.04, 1.04]),
        "episode": np.array([0, 0, 1]),
        "final_required_tcp_local_m": np.zeros((3, 3)),
        "tcp_derived_local_m": np.array([[0.03, 0.0, 0.0], [0.001, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        "processed_tcp_target_local_m": np.zeros((3, 3)),
        "raw_normalized_mean": np.zeros((3, 3)),
        "raw_absolute_tcp_request_local_m": np.zeros((3, 3)),
        "clipped_tcp_request_local_m": np.zeros((3, 3)),
        "body_qpos_local_m": np.zeros((3, 3)),
        "body_goal_local_m": np.zeros((3, 3)),
        "body_signed_error_m": np.zeros((3, 3)),
    }
    rows = audit_module.summarize_trace(trace)
    assert len(rows) == 3
    assert rows[0]["action_time_bin_s"] == [0.0, 1.0]
    assert rows[0]["derived_tcp_tracking"]["median_norm_mm"] == 30.0
    assert rows[1]["action_time_bin_s"] == [1.0, 1.5]
    assert rows[1]["derived_tcp_tracking"]["median_norm_mm"] == 1.0


def test_accessibility_probe_prespecified_seed_folds_and_label_order():
    rng = np.random.default_rng(5)
    variants = np.tile(np.arange(4), 16)
    socket = rng.uniform(0.0, 0.05, size=(64, 3))
    features = np.column_stack([socket[:, :2], np.eye(4)[variants]])
    inspection = {
        "variant_id": variants,
        "socket_goal_local_m": socket,
        "reset_seed": np.repeat(audit_module.SEEDS, 16),
        "cnn_wrist": features[:, :3],
        "cnn_external": features[:, 3:],
        "gru": features.copy(),
        "proprio": rng.normal(size=(64, 2)),
    }
    result = audit_module.accessibility_probe(inspection, ("wrist", "external"))
    assert result["alpha"] == 0.001
    assert len(result["folds"]) == 16
    assert {fold["features"] for fold in result["folds"]} == {
        "cnn_only",
        "proprio_only",
        "cnn_plus_proprio",
        "gru_only",
    }
    for fold in result["folds"]:
        assert fold["training_episodes"] == 48
        assert fold["metrics"]["episodes"] == 16
        if fold["features"] in ("cnn_only", "gru_only"):
            assert fold["metrics"]["variant_accuracy"] == 1.0
            assert fold["metrics"]["socket_xy_p90_error_mm"] < 0.01
