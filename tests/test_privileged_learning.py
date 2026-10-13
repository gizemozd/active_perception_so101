"""State-only diagnosis must change information access without changing task rewards."""

import json
from dataclasses import asdict, replace

import pytest
import torch
from mjlab.rl import RslRlVecEnvWrapper

from active_perception_arms.config import Experiment, saved_experiment
from active_perception_arms.environment import make_env
from active_perception_arms.evaluate import load_actor
from active_perception_arms.policy import CompactPPO, runner_cfg
from active_perception_arms.train import experiment_from_args, parser


@pytest.mark.parametrize("task", ["transfer", "push"])
def test_privileged_policy_updates_and_roundtrips_without_images(task, tmp_path):
    torch.set_num_threads(1)
    cfg = Experiment(
        task=task,
        num_envs=8,
        actor_observation="privileged",
        memory="none",
        task_revision="pick_place_v2" if task == "transfer" else "push_v2",
        reward_profile=task + "_dense",
        manip_control="cartesian",
        condition="wrist_static",
    )
    env = make_env(cfg, "cpu")
    try:
        wrapped = RslRlVecEnvWrapper(env)
        obs, _ = wrapped.reset()
        assert cfg.sensors == () and not cfg.render_sensors
        assert set(obs.keys()) == {"proprio", "critic", "privileged"}
        assert all(t.dtype == torch.float32 for t in obs.values())
        options = asdict(runner_cfg(cfg, 2))
        options["multi_gpu"] = None
        for name in ("actor", "critic"):
            options[name] = {k: v for k, v in options[name].items() if v is not None}
        alg = CompactPPO.construct_algorithm(obs, wrapped, options, "cpu")
        assert not alg.actor.is_recurrent and not hasattr(alg.actor, "cnns")
        assert tuple(alg.actor.obs_groups) == ("privileged",)
        before = {k: v.clone() for k, v in alg.actor.state_dict().items() if "mlp" in k}
        with torch.inference_mode():
            for _ in range(24):
                action = alg.act(obs)
                obs, reward, dones, info = wrapped.step(action)
                alg.process_env_step(obs, reward, dones, info)
            alg.compute_returns(obs)
        metrics = alg.update()
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        assert any(not torch.equal(v, alg.actor.state_dict()[k]) for k, v in before.items())
        (tmp_path / "experiment.json").write_text(json.dumps(cfg.to_dict()))
        (tmp_path / "runner.json").write_text(json.dumps(options))
        torch.save({"actor_state_dict": alg.actor.state_dict()}, tmp_path / "model.pt")
        restored = load_actor(tmp_path / "model.pt", obs, "cpu")
        with torch.inference_mode():
            torch.testing.assert_close(restored(obs), alg.actor(obs))
    finally:
        env.close()


@pytest.mark.parametrize("task", ["transfer", "push"])
def test_state_observations_do_not_change_physics_or_reward(task):
    torch.set_num_threads(1)
    cfg = Experiment(
        task=task,
        num_envs=2,
        render_sensors=False,
        initial_seconds=0,
        task_revision="pick_place_v2" if task == "transfer" else "push_v2",
        reward_profile=task + "_dense",
        manip_control="cartesian",
        condition="wrist_static",
    )
    snapshots = []
    for observation in ("rgb", "privileged"):
        env = make_env(replace(cfg, actor_observation=observation), "cpu")
        try:
            env.reset(seed=333)
            sequence = []
            for _ in range(12):
                _, reward, *_ = env.step(torch.zeros(2, cfg.action_dim))
                sequence.append((env.sim.data.qpos.clone(), reward.clone()))
            snapshots.append(sequence)
        finally:
            env.close()
    for (q0, r0), (q1, r1) in zip(*snapshots, strict=True):
        torch.testing.assert_close(q0, q1, rtol=0, atol=0)
        torch.testing.assert_close(r0, r1, rtol=0, atol=0)


def test_privilege_cannot_be_enabled_by_resume_accident(tmp_path):
    cfg = Experiment(task="transfer", condition="wrist_static")
    assert saved_experiment({"task": "transfer"}).actor_observation == "rgb"
    (tmp_path / "experiment.json").write_text(json.dumps(cfg.to_dict()))
    args = parser().parse_args(
        [
            "--task",
            "transfer",
            "--resume",
            str(tmp_path / "model.pt"),
            "--actor-observation",
            "privileged",
        ]
    )
    with pytest.raises(ValueError, match="actor_observation"):
        experiment_from_args(args)
