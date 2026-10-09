"""The value bootstrap reads the next observation without consuming its memory."""

import copy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel, RNNModel
from tensordict import TensorDict

from active_perception_arms.policy import CompactPPO


def hidden_copy(model):
    state = model.get_hidden_state()
    if isinstance(state, tuple):
        return tuple(part.clone() for part in state)
    return None if state is None else state.clone()


def assert_same_hidden(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))
    else:
        assert torch.equal(actual, expected)


def algorithm(algorithm_type, critic):
    # Exercise the production GAE calculation without constructing an optimizer.
    result = algorithm_type.__new__(algorithm_type)
    result.critic = critic
    result.gamma, result.lam = 0.99, 0.95
    result.normalize_advantage_per_mini_batch = False
    result.storage = SimpleNamespace(
        num_transitions_per_env=3,
        values=torch.tensor([[[0.2], [-0.3]], [[0.4], [-0.1]], [[0.8], [0.0]]]),
        rewards=torch.tensor([[[0.1], [0.2]], [[-0.1], [0.4]], [[0.5], [0.1]]]),
        dones=torch.tensor([[[False], [False]], [[False], [True]], [[False], [False]]]),
        returns=torch.zeros(3, 2, 1),
    )
    return result


@pytest.mark.parametrize(
    ("model_type", "warmed"),
    [("gru", True), ("lstm", True), ("gru", False), ("lstm", False), ("mlp", False)],
)
def test_bootstrap_preserves_memory_and_upstream_return_targets(model_type, warmed):
    torch.manual_seed(41)
    previous = TensorDict({"critic": torch.tensor([[0.1, -0.2], [0.7, 0.3]])}, [2])
    next_obs = TensorDict({"critic": torch.tensor([[0.5, 0.4], [-0.3, 0.9]])}, [2])
    options = dict(hidden_dims=[4], obs_normalization=False)
    if model_type == "mlp":
        critic = MLPModel(previous, {"critic": ["critic"]}, "critic", 1, **options)
    else:
        critic = RNNModel(
            previous,
            {"critic": ["critic"]},
            "critic",
            1,
            rnn_type=model_type,
            rnn_hidden_dim=4,
            **options,
        )
    with torch.inference_mode():
        if warmed:
            critic(previous)
        before = hidden_copy(critic)
        reference_critic = copy.deepcopy(critic)
        baseline = algorithm(PPO, copy.deepcopy(critic))
        corrected = algorithm(CompactPPO, critic)
        baseline.compute_returns(next_obs)
        corrected.compute_returns(next_obs)

        assert_same_hidden(critic.get_hidden_state(), before)
        assert torch.equal(corrected.storage.returns, baseline.storage.returns)
        assert torch.equal(corrected.storage.advantages, baseline.storage.advantages)
        # The first value of the next rollout must process this observation once.
        next_value = critic(next_obs)
        assert torch.equal(next_value, reference_critic(next_obs))
        assert_same_hidden(critic.get_hidden_state(), reference_critic.get_hidden_state())
        if model_type != "mlp":
            assert not torch.equal(next_value, baseline.critic(next_obs))


def test_failed_return_computation_also_restores_critic_memory():
    torch.manual_seed(7)
    obs = TensorDict({"critic": torch.tensor([[0.1], [0.7]])}, [2])
    critic = RNNModel(
        obs,
        {"critic": ["critic"]},
        "critic",
        1,
        hidden_dims=[4],
        rnn_type="gru",
        rnn_hidden_dim=4,
    )
    with torch.inference_mode():
        critic(obs)
        before = hidden_copy(critic)
        corrected = algorithm(CompactPPO, critic)
        # Fail after the upstream bootstrap has consumed an observation.
        corrected.storage.rewards = torch.zeros(0, 2, 1)
        with pytest.raises(IndexError):
            corrected.compute_returns(obs)
        assert_same_hidden(critic.get_hidden_state(), before)
