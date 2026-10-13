"""Insertion reward, historical stage shaping and precision control.

Stage history is privileged reward/critic state. Neither stages nor true target
positions are actor observations. Geometry and physical success remain unchanged.
"""

import torch

STEP_METERS = 0.005
SUCCESS_BONUS = 100.0


def insertion_reward(obj, goal, quat=None):
    """Port of active_perception_arms/tasks/insertion/mdp/rewards.py.

    Upstream insertion_reward uses -log(100 * error + 1), REPLACED by
    1000 at instantaneous position error <2 mm. No time/action/stage costs.
    Optional rotation uses the upstream quaternion Euclidean error against
    identity; it affects shaping only, not the position-based reward bonus.
    The caller supplies current physical poses and the variant-corrected goal.
    Episode termination remains the separately audited three-sample hold.
    """
    from .plug import SUCCESS_DISTANCE

    dpos = obj - goal
    err_sq = torch.sum(dpos * dpos, dim=-1)
    if quat is not None:
        target_quat = torch.zeros_like(quat)
        target_quat[:, 0] = 1.0
        dquat = torch.linalg.norm(quat - target_quat, dim=-1)
        err_sq = err_sq + dquat * dquat
    reward = -torch.log(100.0 * torch.sqrt(err_sq) + 1.0)
    instantaneous = torch.linalg.norm(dpos, dim=-1) < SUCCESS_DISTANCE
    return torch.where(instantaneous, torch.full_like(reward, 1000.0), reward)


def update_stages(tcp, obj, goal, raised, aligned):
    """Use current physical positions, expressed relative to each world's origin."""
    raised |= tcp[:, 2] >= 0.087
    xy = (obj[:, :2] - goal[:, :2]).norm(dim=-1)
    aligned |= raised & (tcp[:, 2] >= 0.084) & (xy < 0.003)


def density(tcp, obj, goal, raised, aligned):
    """Bounded dense stages: lift [0,1], align [1,2], insert [1,3].

    Retaining alignment history cannot collect insertion credit after drifting:
    both terms continue to depend on current XY error. A successful terminal
    bonus exceeds the discounted maximum remaining dense reward (0.6 / .01).
    """
    xy = (obj[:, :2] - goal[:, :2]).norm(dim=-1)
    z = (obj[:, 2] - goal[:, 2]).abs()
    lift = ((tcp[:, 2] - 0.080) / 0.009).clamp(0, 1)
    clearance = ((tcp[:, 2] - 0.083) / 0.005).clamp(0, 1)
    approach = 1 + clearance * torch.exp(-xy / 0.015)
    insertion = 1 + torch.exp(-xy / 0.015) + torch.exp(-xy / 0.003 - z / 0.008)
    return torch.where(aligned, insertion, torch.where(raised, approach, lift))


def features(env, state, tcp, obj, goal):
    local_tcp = tcp - env.scene.env_origins
    local_obj = obj - env.scene.env_origins
    local_goal = goal - env.scene.env_origins
    update_stages(local_tcp, local_obj, local_goal, state.plug_raised, state.plug_aligned)
    return density(local_tcp, local_obj, local_goal, state.plug_raised, state.plug_aligned)
