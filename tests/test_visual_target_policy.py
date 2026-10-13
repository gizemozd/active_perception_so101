"""The structured insertion controller uses only policy observations at deployment."""

import torch

from active_perception_arms.visual_target_policy import VisualTargetPolicy


def test_motion_plan_retains_visual_estimate_and_uses_no_privileged_keys():
    # Exercise the physical action decoder without constructing/training a CNN.
    actor = object.__new__(VisualTargetPolicy)
    torch.nn.Module.__init__(actor)
    actor.episode_seconds = 8
    actor.start_tcp = actor.estimated_target = None
    actor.action_low = torch.tensor([-0.07, -0.045, 0.055])
    actor.action_high = torch.tensor([0.07, 0.076, 0.09])
    obs = {"proprio": torch.zeros(2, 51)}
    obs["proprio"][:, 36:39] = torch.tensor([0.0, 0.02, 0.08])
    target = torch.tensor([[0.025, 0.07, 0.061], [0.025, 0.04, 0.061]])
    prediction = 2 * (target - actor.action_low) / (actor.action_high - actor.action_low) - 1
    action = actor.action_from_target(prediction, obs)
    physical = actor.action_low + (action + 1) * 0.5 * (actor.action_high - actor.action_low)
    torch.testing.assert_close(physical[:, :2], obs["proprio"][:, 36:38])
    torch.testing.assert_close(physical[:, 2], torch.full((2,), 0.09))
    obs["proprio"][:, -1] = 2 / 8
    action = actor.action_from_target(prediction, obs)
    physical = actor.action_low + (action + 1) * 0.5 * (actor.action_high - actor.action_low)
    torch.testing.assert_close(physical[:, :2], target[:, :2])
    obs["proprio"][:, -1] = 3 / 8
    # Once inspection ends, arbitrary subsequent visual errors cannot move the target.
    action = actor.action_from_target(torch.zeros_like(prediction), obs)
    physical = actor.action_low + (action + 1) * 0.5 * (actor.action_high - actor.action_low)
    torch.testing.assert_close(physical[:, :2], target[:, :2])
    torch.testing.assert_close(physical[:, 2], torch.full((2,), 0.079))
