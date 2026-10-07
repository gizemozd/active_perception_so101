"""Regression for the installed RSL-RL/W&B writer compatibility boundary."""

import wandb

from active_perception_arms.telemetry import compatible_wandb_writer


def test_native_wandb_writer_works_with_current_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    writer = compatible_wandb_writer(tmp_path, {"wandb_project": "unit-test-no-upload"})
    try:
        assert wandb.run is not None
        writer.add_scalar("train/transitions", 24, 0)
        writer.add_scalar("train/reward", 0.1, 0)
        writer.flush()
        assert list(tmp_path.glob("events.out.tfevents.*"))
    finally:
        writer.close()
        writer.stop()


def test_resume_can_extend_wandb_budget(tmp_path, monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class Environment:
        num_envs: int = 512

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_RESUME", "must")
    writer = compatible_wandb_writer(tmp_path, {"wandb_project": "unit-test-no-upload"})
    try:
        writer.store_config(Environment(), {"max_iterations": 100})
        writer.store_config(Environment(), {"max_iterations": 1500})
        assert wandb.config["train_cfg"]["max_iterations"] == 1500
        assert wandb.config["env_cfg"]["num_envs"] == 512
    finally:
        writer.close()
        writer.stop()


def test_resume_restores_adaptive_lr_and_next_iteration(tmp_path):
    from types import SimpleNamespace

    import torch
    from rsl_rl.algorithms import PPO

    from active_perception_arms.train import restore_training_state

    parameter = torch.nn.Parameter(torch.ones(1))
    saved_optimizer = torch.optim.Adam([parameter], lr=7.59375e-5)
    (parameter**2).sum().backward()
    saved_optimizer.step()
    checkpoint = tmp_path / "model_99.pt"
    torch.save({"optimizer_state_dict": saved_optimizer.state_dict(), "iter": 99}, checkpoint)
    algorithm = SimpleNamespace(
        optimizer=torch.optim.Adam([parameter], lr=3e-4), learning_rate=3e-4, rnd=None
    )
    runner = SimpleNamespace(alg=algorithm, logger=SimpleNamespace(tot_timesteps=0))

    def load(path):
        data = torch.load(path, weights_only=True)
        PPO.load(algorithm, data, {"optimizer": True, "iteration": True}, strict=True)
        runner.current_learning_iteration = data["iter"]

    runner.load = load
    restore_training_state(runner, checkpoint, 512)
    assert runner.current_learning_iteration == 100
    assert runner.logger.tot_timesteps == 1_228_800
    assert algorithm.learning_rate == 7.59375e-5
    assert algorithm.optimizer.state[parameter]["step"].item() == 1


def test_episode_metrics_survive_rollout_starting_without_reset(tmp_path):
    import torch
    from rsl_rl.utils.logger import Logger

    from active_perception_arms.episode_logging import install_episode_logging

    class Capture:
        def __init__(self):
            self.values = {}

        def add_scalar(self, key, value, step):
            self.values[key] = float(value)

    logger = Logger(
        str(tmp_path),
        {"algorithm": {"rnd_cfg": None}, "num_steps_per_env": 3},
        {},
        2,
        False,
        1,
        0,
        "cpu",
    )
    logger.writer = Capture()
    logger.logger_type = "wandb"
    install_episode_logging(logger)
    reward = torch.ones(2)
    logger.process_env_step(reward, torch.zeros(2), {"log": {}})
    logger.process_env_step(
        reward, torch.ones(2), {"log": {"Episode_Metrics/success_rate": torch.tensor(0.5)}}
    )
    logger.process_env_step(reward, torch.zeros(2), {"log": {}})
    logger.log(
        it=0,
        start_it=0,
        total_it=1,
        collect_time=1,
        learn_time=1,
        loss_dict={},
        learning_rate=0.001,
        action_std=torch.ones(2),
        rnd_weight=None,
    )
    assert logger.writer.values["Episode_Metrics/success_rate"] == 0.5
    assert logger.writer.values["Train/mean_reward"] == 2.0
