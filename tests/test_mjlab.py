from dataclasses import asdict

import pytest
import torch

from active_perception_arms import mdp
from active_perception_arms.config import Experiment
from active_perception_arms.environment import make_env


@pytest.fixture(scope="module")
def env():
    e = make_env(
        Experiment(
            task="plug",
            condition="initial",
            num_envs=2,
            width=32,
            height=32,
            occlusion="phase",
            randomize=False,
        ),
        device="cpu",
    )
    yield e
    e.close()


def test_warp_rgb_actor_runner_and_subset_reset(env):
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from active_perception_arms.policy import runner_cfg

    obs, _ = env.reset(seed=12)
    assert obs["proprio"].shape == (2, 43)
    assert obs["critic"].shape == (2, 68)
    for name in ("wrist", "external"):
        assert obs[name].shape == (2, 3, 32, 32)
        assert obs[name].dtype == torch.uint8
        assert obs[name].max() > obs[name].min()
    options = asdict(runner_cfg(Experiment(condition="initial", num_envs=2, width=32, height=32)))
    options["num_steps_per_env"] = 2
    options["algorithm"]["num_mini_batches"] = 1
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), options, device="cpu")
    with torch.inference_mode():
        actions = runner.alg.actor(obs)
    obs, rewards, _, _, _ = env.step(actions)
    assert torch.isfinite(rewards).all()
    assert all(torch.isfinite(value).all() for value in obs.values())
    other = env.scene["object"].data.root_link_pos_w[1].clone()
    env.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(other, env.scene["object"].data.root_link_pos_w[1])
    torch.testing.assert_close(
        env.action_manager.get_term("arms").targets["manipulator"][0],
        mdp.state(env).home["manipulator"],
    )


def test_camera_freeze_and_no_actor_privileged_state_leak(env):
    env.reset()
    term = env.action_manager.get_term("arms")
    original = mdp.proprio(env).clone()
    mdp.state(env).onset[:] += 0.5
    mdp.state(env).duration[:] += 0.7
    torch.testing.assert_close(original, mdp.proprio(env))
    action = torch.ones(2, 10) * 0.1
    before = term.targets["camera_arm"].clone()
    term.process_actions(action)
    assert not torch.equal(before, term.targets["camera_arm"])
    env.episode_length_buf[:] = 30
    before = term.targets["camera_arm"].clone()
    term.process_actions(action)
    torch.testing.assert_close(before, term.targets["camera_arm"])
    env.reset()


def test_optical_panel_updates_and_success_requires_hold(env):
    env.reset()
    mdp.update_panel(env, torch.full((2,), 4.0))
    model = env.sim.mj_model
    index = model.body("occluder/panel").mocapid[0]
    local = env.sim.data.mocap_pos[:, index] - env.scene.env_origins
    assert (local[:, 2] > 0).all()
    mdp.update_panel(env, torch.zeros(2))
    assert (env.sim.data.mocap_pos[:, index, 2] - env.scene.env_origins[:, 2] < 0).all()
    assert not mdp.success(env).any()


@pytest.mark.gpu
@pytest.mark.parametrize("task", ("plug", "transfer", "push"))
def test_cuda_graph_cameras_and_runner(task):
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA GPU unavailable on this host")
    cfg = Experiment(task=task, num_envs=8, width=32, height=32)
    gpu = make_env(cfg, "cuda:0")
    try:
        obs, _ = gpu.reset()
        for _ in range(4):
            obs, reward, _, _, _ = gpu.step(torch.zeros(8, cfg.action_dim, device="cuda"))
        torch.cuda.synchronize()
        assert obs["external"].is_cuda and obs["external"].dtype == torch.uint8
        assert torch.isfinite(reward).all()
    finally:
        gpu.close()


def test_terminal_success_survives_auto_reset(env, monkeypatch):
    env.reset()
    s = mdp.state(env)
    s.hold[:] = 2
    s.last_success_step = -1
    monkeypatch.setattr(mdp, "instantaneous_success", lambda _: torch.ones(2, dtype=torch.bool))
    _, reward, terminated, _, _ = env.step(torch.zeros(2, 10))
    assert terminated.all()
    assert env.termination_manager.get_term("success").all()
    assert (reward > 9).all()
    assert (env.episode_length_buf == 0).all()
    assert not s.succeeded.any()


@pytest.mark.parametrize(
    "condition,keys",
    [("wrist", {"wrist"}), ("static", {"external"}), ("wrist_static", {"wrist", "external"})],
)
def test_requested_sensor_configuration(condition, keys):
    from active_perception_arms.environment import make_env_cfg

    cfg = make_env_cfg(Experiment(condition=condition, num_envs=1))
    assert {sensor.name for sensor in cfg.scene.sensors} == keys
    assert cfg.is_finite_horizon
