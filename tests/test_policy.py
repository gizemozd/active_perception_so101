from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from active_perception_arms.config import Experiment
from active_perception_arms.policy import CompactPPO, runner_cfg


@pytest.mark.parametrize("memory", ["gru", "none"])
def test_recurrent_rollout_storage_and_padded_backward(memory):
    """Exercise the real RSL interface without learning or an optimizer step."""
    exp = Experiment(num_envs=4, width=32, height=32, memory=memory)
    obs = TensorDict(
        {
            "proprio": torch.randn(4, 43),
            "critic": torch.randn(4, 68),
            "wrist": torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8),
            "external": torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8),
        },
        [4],
    )
    cfg = asdict(runner_cfg(exp))
    cfg["multi_gpu"] = None
    cfg["num_steps_per_env"] = 4
    cfg["algorithm"]["num_mini_batches"] = 2
    # MjlabOnPolicyRunner strips optional None model fields before construction.
    for name in ("actor", "critic"):
        cfg[name] = {k: v for k, v in cfg[name].items() if v is not None}
    alg = CompactPPO.construct_algorithm(
        obs, SimpleNamespace(num_envs=4, num_actions=10), cfg, "cpu"
    )
    before = {k: v.clone() for k, v in alg.actor.named_parameters()}
    with torch.inference_mode():
        for step in range(4):
            assert alg.act(obs).shape == (4, 10)
            dones = torch.tensor([step == 1, step == 2, False, step == 3])
            alg.process_env_step(obs, torch.zeros(4), dones, {})
    assert alg.storage.observations["wrist"].dtype == torch.uint8
    assert torch.equal(alg.storage.observations["wrist"][0], obs["wrist"])
    batch = next(alg.storage.recurrent_mini_batch_generator(2, 1))
    out = alg.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0])
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in alg.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in alg.actor.cnns.parameters())
    assert all(torch.equal(before[k], v) for k, v in alg.actor.named_parameters())
    with torch.inference_mode():
        alg.actor.reset()
        first = alg.actor(obs)
        second = alg.actor(obs)
        alg.actor.reset()
        assert torch.allclose(first, alg.actor(obs))
        if memory == "gru":
            assert not torch.allclose(first, second)
        else:
            assert torch.allclose(first, second)
