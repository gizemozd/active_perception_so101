"""Physical yaw control, truthful scoring and the recovered FF training recipe."""

import json
import math
from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from active_perception_arms import mdp, plug
from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.evaluate import load_actor
from active_perception_arms.native import NativeEnv, ScriptedPolicy
from active_perception_arms.policy import CompactPPO, runner_cfg
from active_perception_arms.train import main


def test_wrist_yaw_moves_physical_plug_and_returns_to_aligned_grasp():
    torch.set_num_threads(1)
    env = NativeEnv(
        Experiment(
            condition="wrist_static", plug_wrist_yaw=True, render_sensors=False, randomize=False
        )
    )
    action = np.zeros(4)
    action[:3] = (
        2 * (env.tcp_target - plug.ACTION_LOW) / (np.array(plug.ACTION_HIGH) - plug.ACTION_LOW) - 1
    )
    action[3] = 1
    for _ in range(25):
        env.step(action)
    assert env.wrist_yaw_target == 0  # common inspection pause
    env.step(action)
    assert env.wrist_yaw_target == pytest.approx(plug.WRIST_YAW_MAX_DELTA)
    for _ in range(19):
        env.step(action)
    quat = env.data.qpos[env.object_adr + 3 : env.object_adr + 7]
    yaw = 2 * np.arctan2(quat[3], quat[0])
    assert yaw == pytest.approx(plug.WRIST_YAW_LIMIT, abs=0.025)
    assert np.linalg.norm(env.tcp - env.tcp_target) < 0.002
    action[3] = 0
    for _ in range(20):
        env.step(action)
    quat = env.data.qpos[env.object_adr + 3 : env.object_adr + 7]
    assert 2 * np.arctan2(np.linalg.norm(quat[1:]), abs(quat[0])) < plug.SUCCESS_ANGLE


@pytest.mark.parametrize("variant", plug.VARIANTS)
def test_rotated_body_at_old_success_position_is_not_success(variant, monkeypatch):
    cfg = Experiment(
        plug_variant=variant, plug_wrist_yaw=True, randomize=False, render_sensors=False
    )
    native = NativeEnv(cfg)
    adr = native.object_adr
    native.data.qpos[adr : adr + 7] = [*native.goal, math.cos(0.15), 0, 0, math.sin(0.15)]
    mujoco.mj_forward(native.model, native.data)
    assert not native.success()
    goal = torch.tensor(native.goal[None], dtype=torch.float32)
    task = SimpleNamespace(cfg=cfg, object_qpos_address=0)
    env = SimpleNamespace(
        task_state=task,
        sim=SimpleNamespace(
            data=SimpleNamespace(
                qpos=torch.tensor(native.data.qpos[None, adr : adr + 7], dtype=torch.float32)
            )
        ),
        scene={
            "object": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=torch.zeros(1, 3)))
        },
    )
    monkeypatch.setattr(mdp, "positions", lambda _: (None, goal, None, goal))
    assert not mdp.instantaneous_success(env).item()
    env.sim.data.qpos[:, 3:] *= -1
    assert not mdp.instantaneous_success(env).item()  # quaternion sign invariant
    env.sim.data.qpos[:, 3:] = torch.tensor([1.0, 0, 0, 0])
    assert mdp.instantaneous_success(env).item()


@pytest.mark.parametrize("variant", plug.VARIANTS)
def test_optional_yaw_preserves_scripted_insertion(variant):
    env = NativeEnv(
        Experiment(plug_variant=variant, plug_wrist_yaw=True, randomize=False, render_sensors=False)
    )
    controller = ScriptedPolicy(env)
    held = 0
    for _ in range(88):
        held = held + 1 if env.step(controller()) else 0
        if held == 3:
            break
    assert held == 3
    assert env.cfg.action_dim == 9


def test_legacy_ff_performs_finite_ppo_update_through_both_image_encoders(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    exp = Experiment(
        memory="none",
        policy_profile="legacy_ff",
        plug_wrist_yaw=True,
        num_envs=8,
        width=32,
        height=32,
    )
    obs = TensorDict(
        {
            "proprio": torch.randn(8, 52),
            "critic": torch.randn(8, 77),
            "wrist": torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8),
            "external": torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8),
        },
        [8],
    )
    cfg = asdict(runner_cfg(exp))
    assert cfg["actor"]["hidden_dims"] == (256, 256, 128)
    assert cfg["algorithm"]["num_learning_epochs"] == 5
    assert cfg["algorithm"]["num_mini_batches"] == 4
    cfg.update(multi_gpu=None, num_steps_per_env=4)
    for name in ("actor", "critic"):
        cfg[name] = {k: v for k, v in cfg[name].items() if v is not None}
    alg = CompactPPO.construct_algorithm(
        obs, SimpleNamespace(num_envs=8, num_actions=9), cfg, "cpu"
    )
    assert not alg.actor.is_recurrent and not alg.critic.is_recurrent
    before = {k: v.clone() for k, v in alg.actor.cnns.state_dict().items()}
    with torch.inference_mode():
        for step in range(4):
            assert alg.act(obs).shape == (8, 9)
            alg.process_env_step(
                obs,
                torch.arange(8, dtype=torch.float32) / 8,
                torch.tensor([step == 2] + [False] * 7),
                {},
            )
        alg.compute_returns(obs)
    metrics = alg.update()
    assert all(math.isfinite(v) for v in metrics.values())
    for camera in exp.sensors:
        assert any(
            not torch.equal(v, before[k])
            for k, v in alg.actor.cnns.state_dict().items()
            if k.startswith(camera)
        )
    assert cfg["actor"]["distribution_cfg"]["class_name"] == "GaussianDistribution"
    (tmp_path / "runner.json").write_text(json.dumps(cfg))
    (tmp_path / "experiment.json").write_text(json.dumps(exp.to_dict()))
    checkpoint = tmp_path / "model.pt"
    torch.save({"actor_state_dict": alg.actor.state_dict()}, checkpoint)
    restored = load_actor(checkpoint, obs, "cpu")
    with torch.inference_mode():
        torch.testing.assert_close(restored(obs), alg.actor(obs))


def test_warp_yaw_camera_slices_and_partial_reset():
    from mjlab.envs import ManagerBasedRlEnv

    from active_perception_arms.environment import make_env_cfg

    cfg = Experiment(num_envs=4, plug_wrist_yaw=True, render_sensors=False, randomize=False)
    setup = make_env_cfg(cfg)
    setup.terminations = {}
    env = ManagerBasedRlEnv(setup, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["proprio"].shape == (4, 52)
        term = env.action_manager.get_term("arms")
        env.episode_length_buf[:] = 25
        actions = torch.zeros(4, 9)
        actions[:, :3] = (
            2 * (term.tcp_target - term.plug_ik.low) / (term.plug_ik.high - term.plug_ik.low) - 1
        )
        actions[:, 3] = 1
        before_camera = term.gimbal_target.clone()
        env.step(actions)
        torch.testing.assert_close(
            term.wrist_yaw_target, torch.full((4,), plug.WRIST_YAW_MAX_DELTA)
        )
        torch.testing.assert_close(term.gimbal_target, before_camera)
        actions[:, 4] = 1
        env.step(actions)
        assert torch.all(term.gimbal_target[:, 0] > before_camera[:, 0])
        preserved = term.wrist_yaw_target[1:].clone()
        mdp.reset_task(env, torch.tensor([0]), cfg)
        assert term.wrist_yaw_target[0] == 0
        torch.testing.assert_close(term.wrist_yaw_target[1:], preserved)
    finally:
        env.close()


def test_new_settings_resume_and_old_checkpoint_defaults(tmp_path, capsys):
    original = Experiment().to_dict()
    original.pop("plug_wrist_yaw")
    original.pop("policy_profile")
    restored = saved_experiment(original)
    assert restored.action_dim == 8 and restored.policy_profile == "compact"
    cfg = Experiment(
        memory="none",
        policy_profile="legacy_ff",
        plug_wrist_yaw=True,
        reward_profile="legacy_log_hold",
    )
    (tmp_path / "experiment.json").write_text(json.dumps(cfg.to_dict()))
    (tmp_path / "runner.json").write_text(json.dumps(asdict(runner_cfg(cfg))))
    args = ["--task", "plug", "--resume", str(tmp_path / "model_99.pt"), "--dry-run"]
    main(args)
    output = json.loads(capsys.readouterr().out)
    assert output["initial_std"] == 1 and output["experiment"]["plug_wrist_yaw"]
    assert output["experiment"]["memory"] == "none"
    assert output["optimization"]["learning_rate"] == 0.001
    with pytest.raises(ValueError, match="plug_wrist_yaw differs"):
        main([*args, "--no-plug-wrist-yaw"])
    with pytest.raises(ValueError, match="policy_profile differs"):
        main([*args, "--policy-profile", "compact"])


def test_orientation_audit_retains_all_three_physical_samples():
    from active_perception_arms.evaluation_audit import summarize_termination_audit

    record = dict(
        success=True,
        failure=False,
        timeout=False,
        success_hold_steps=3,
        manager_position_error_m=0.001,
        qpos_position_error_m=0.001,
        derived_to_qpos_displacement_m=0.0,
        last_three_manager_errors_m=[0.001] * 3,
        last_three_orientation_errors_rad=[0.0, 0.1, 0.0],
    )
    assert summarize_termination_audit([record], "plug")["success_orientation_violations"] == 1
    record["last_three_orientation_errors_rad"] = [0.0, 0.01, 0.0]
    assert summarize_termination_audit([record], "plug")["success_orientation_violations"] == 0
