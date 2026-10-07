"""Search diagnostics must preserve native PPO updates, including recurrent batches."""

import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from active_perception_arms.config import Experiment
from active_perception_arms.optimization_diagnostics import install_optimization_diagnostics
from active_perception_arms.policy import CompactPPO, runner_cfg


@pytest.mark.parametrize("schedule", ["adaptive", "fixed"])
def test_diagnostics_preserve_update_and_weight_episode_counts(schedule):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    obs = TensorDict(
        {
            "proprio": torch.randn(4, 51),
            "critic": torch.randn(4, 76),
            "wrist": torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8),
        },
        [4],
    )
    cfg = asdict(runner_cfg(Experiment(condition="wrist", width=32, height=32)))
    cfg.update(multi_gpu=None, num_steps_per_env=4)
    cfg["algorithm"].update(num_mini_batches=2, num_learning_epochs=1, schedule=schedule)
    for name in ("actor", "critic"):
        cfg[name] = {k: v for k, v in cfg[name].items() if v is not None}
    alg = CompactPPO.construct_algorithm(
        obs, SimpleNamespace(num_envs=4, num_actions=3), cfg, "cpu"
    )
    with torch.inference_mode():
        for i in range(4):
            alg.act(obs)
            alg.process_env_step(
                obs, torch.ones(4), torch.tensor([i == 1, False, False, False]), {}
            )
        alg.compute_returns(obs)
    observed = copy.deepcopy(alg)
    logger = SimpleNamespace(process_env_step=lambda *args: None)
    install_optimization_diagnostics(observed, logger)
    # One success among two resets, then one success among four resets: 2/6,
    # whereas the unweighted mean of the two batch rates is 0.375.
    for dones, rate in [(torch.tensor([1, 1, 0, 0]), 0.5), (torch.ones(4), 0.25)]:
        logger.process_env_step(
            torch.ones(4),
            dones,
            {
                "log": {"Episode_Metrics/success_rate": torch.tensor(rate)},
            },
        )
    rng = torch.get_rng_state()
    expected = alg.update()
    torch.set_rng_state(rng)
    actual = observed.update()
    for key in expected:
        assert actual[key] == expected[key]
    for p, q in zip(alg.actor.parameters(), observed.actor.parameters(), strict=True):
        assert torch.equal(p, q)
    for p, q in zip(alg.critic.parameters(), observed.critic.parameters(), strict=True):
        assert torch.equal(p, q)
    assert actual["diagnostic_episode_weighted_success"] == pytest.approx(1 / 3)
    assert actual["diagnostic_completed_episodes"] == 6
    assert actual["diagnostic_kl"] >= 0
    assert 0 <= actual["diagnostic_clip_fraction"] <= 1
    assert all(actual[f"diagnostic_action_std_{i}"] > 0 for i in range(3))
