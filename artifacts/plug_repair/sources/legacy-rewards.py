"""Reward terms for the insertion task."""

from __future__ import annotations

import math

import torch

from active_perception_arms.tasks.insertion.mdp.state import (
  INSERTION_SUCCESS_DIST,
  get_insertion_state,
)


def insertion_reward(env, *, with_rotation: bool) -> torch.Tensor:
  state = get_insertion_state(env)
  obj_pose = env.scene["object"].data.root_link_pose_w
  pos = obj_pose[:, :3]
  quat = obj_pose[:, 3:7]
  dpos = pos - state.target_pos
  dquat = torch.linalg.norm(quat - state.target_quat, dim=-1)
  err_sq = torch.sum(dpos * dpos, dim=-1)
  if with_rotation:
    err_sq = err_sq + dquat * dquat
  reward = -torch.log(100.0 * torch.sqrt(err_sq) + 1.0)
  success = torch.linalg.norm(dpos, dim=-1) < INSERTION_SUCCESS_DIST
  return torch.where(success, torch.full_like(reward, 1000.0), reward)


def camera_action_smoothness_penalty(
  env, *, jerk_weight: float = 0.25
) -> torch.Tensor:
  """Negative camera-command variation cost in policy-action units.

  The first-difference term discourages command chatter; the second-difference
  term discourages abrupt changes in that command velocity.  It is an optional
  reward term (the coefficient is set by the environment configuration), so
  baseline tasks preserve their original objective exactly.
  """
  state = get_insertion_state(env)
  delta_cost = state.camera_action_delta.square().mean(dim=-1)
  jerk_cost = state.camera_action_jerk.square().mean(dim=-1)
  return -(delta_cost + float(jerk_weight) * jerk_cost)


def stack_reward(env) -> torch.Tensor:
  """Stacking shaping: -log distance from block A's origin (bottom-center) to
  block B's LIVE top-center, +1000 on success. B is a free body that can be
  pushed, so the target is recomputed from its current pose every step."""
  from active_perception_arms.robots.so101_constants import (
    BLOCK_B_SIZE,
    STACK_SUCCESS_DIST,
  )

  a_pos = env.scene["object"].data.root_link_pose_w[:, :3]
  b_pos = env.scene["block_b"].data.root_link_pose_w[:, :3]
  target = b_pos.clone()
  target[:, 2] += BLOCK_B_SIZE
  dpos = a_pos - target
  dist = torch.linalg.norm(dpos, dim=-1)
  reward = -torch.log(100.0 * dist + 1.0)
  success = dist < STACK_SUCCESS_DIST
  return torch.where(success, torch.full_like(reward, 1000.0), reward)


def lidbox_reach_term(d: torch.Tensor, d0: float) -> torch.Tensor:
  """Non-negative reach shaping: log(100 d0 + 1) - log(100 d + 1), clipped at
  0 -- exactly 0 at d = d0 and beyond, +log(100 d0 + 1) at the cube."""
  return (math.log(100.0 * d0 + 1.0) - torch.log(100.0 * d + 1.0)).clamp_min(0.0)


def lidbox_reward(env) -> torch.Tensor:
  """Staged lidbox shaping (fixed box yaw); weights in so101_constants:

    r = w_lid clip(theta / theta_open, 0, 1)
      + g_open [w_reach max(0, log(100 d0 + 1) - log(100 d + 1)) + w_grasp 1[captured]]
      + 1[captured] w_lift clip((z_cube - z_floor) / (z_rim - z_floor), 0, 1)
      + B 1[success]

  theta_open = 90 deg (past vertical the free hinge holds the lid on its
  stop), g_open = 1[theta >= theta_open], d = |TCP - cube center| (the reach
  term :func:`lidbox_reach_term` is 0 at and beyond d0 = LIDBOX_REACH_D0, the
  spawn distance, and positive closer in, so opening the lid is never
  penalized), captured =
  the grasp weld engaged by the magnetic capture (events.lidbox_step), z_cube
  = the cube's bottom, z_rim = the wall top. Success (terminations.
  lidbox_success) = captured AND z_cube >= z_rim + LIDBOX_SUCCESS_LIFT.

  lidlatch (``mode.lid_latch``) adds, WHILE THE LATCH IS ACTIVE, a
  press-approach term of the same shape toward the release button:
  1[latched] w_press max(0, log(100 d0 + 1) - log(100 d_TCP->button + 1)),
  w_press = LIDLATCH_REWARD_W_PRESS (its maximum stays below the +1/step lid
  term the released lid pays, so releasing never scores below hovering at the
  button). The lid term itself is unchanged: the spring carries the released
  lid past theta_open and the staged pick terms take over.
  """
  from active_perception_arms.robots.so101_constants import (
    LIDBOX_FLOOR_T,
    LIDBOX_LID_OPEN_ANGLE,
    LIDBOX_REACH_D0,
    LIDBOX_REWARD_W_GRASP,
    LIDBOX_REWARD_W_LID,
    LIDBOX_REWARD_W_LIFT,
    LIDBOX_REWARD_W_REACH,
    LIDBOX_SUCCESS_BONUS,
    LIDBOX_WALL_H,
  )
  from active_perception_arms.tasks.insertion.mdp.events import grasp_weld_active
  from active_perception_arms.tasks.insertion.mdp.state import lidbox_tcp_cube
  from active_perception_arms.tasks.insertion.mdp.terminations import lidbox_success

  tcp, cube, center, lid = lidbox_tcp_cube(env)
  r_lid = LIDBOX_REWARD_W_LID * (lid / LIDBOX_LID_OPEN_ANGLE).clamp(0.0, 1.0)
  g_open = (lid >= LIDBOX_LID_OPEN_ANGLE).float()
  d = torch.linalg.norm(tcp - center, dim=-1)
  r_reach = lidbox_reach_term(d, LIDBOX_REACH_D0)
  captured = grasp_weld_active(env).float()
  lift = ((cube[:, 2] - LIDBOX_FLOOR_T) / (LIDBOX_WALL_H - LIDBOX_FLOOR_T)).clamp(0.0, 1.0)
  reward = (
    r_lid
    + g_open * (LIDBOX_REWARD_W_REACH * r_reach + LIDBOX_REWARD_W_GRASP * captured)
    + captured * LIDBOX_REWARD_W_LIFT * lift
  )
  from active_perception_arms.tasks.insertion.modes import get_mode

  if get_mode(env).lid_latch:
    from active_perception_arms.robots.so101_constants import LIDLATCH_REWARD_W_PRESS
    from active_perception_arms.tasks.insertion.mdp.events import (
      lid_latch_active,
      lidlatch_button_world,
    )

    latched = lid_latch_active(env).float()
    d_btn = torch.linalg.norm(tcp - lidlatch_button_world(env), dim=-1)
    reward = reward + latched * LIDLATCH_REWARD_W_PRESS * lidbox_reach_term(d_btn, LIDBOX_REACH_D0)
  return reward + LIDBOX_SUCCESS_BONUS * lidbox_success(env).float()


def empty_hand_stub_reward(env) -> torch.Tensor:
  """Placeholder for the transfer/lidbox modes (render-approval stage): zero
  reward. Real shaping (magnetic pick, place / lid-open + pick) lands once the
  scenes are approved."""
  return torch.zeros(env.num_envs, device=env.device)


def transfer_reward(env) -> torch.Tensor:
  """Two-stage transfer shaping: approach the cube, then carry it to the
  sampled visual target.  Both potentials are non-negative and zero at their
  corresponding reset-scale distances, avoiding a "do nothing" local optimum.

  The magnetic capture event is the discrete stage transition.  The terminal
  bonus is deliberately the standard 1000 used by the other insertion tasks;
  shaping only supplies dense credit assignment for the pick and transport.
  """
  from active_perception_arms.robots.so101_constants import (
    CUBE_GRIP_Z_OFFSET,
    TRANSFER_REACH_D0,
    TRANSFER_REWARD_W_CAPTURE,
    TRANSFER_REWARD_W_PLACE,
    TRANSFER_REWARD_W_REACH,
    TRANSFER_SUCCESS_BONUS,
  )
  from active_perception_arms.tasks.insertion.mdp.events import grasp_weld_active
  from active_perception_arms.tasks.insertion.mdp.state import manip_tcp_site_index
  from active_perception_arms.tasks.insertion.mdp.terminations import transfer_success

  state = get_insertion_state(env)
  cube = env.scene["object"].data.root_link_pose_w[:, :3]
  tcp = env.scene["manipulator"].data.site_pose_w[:, manip_tcp_site_index(env), :3]
  center = cube.clone()
  center[:, 2] += CUBE_GRIP_Z_OFFSET
  captured = grasp_weld_active(env)

  pick_d = torch.linalg.norm(tcp - center, dim=-1)
  reach = lidbox_reach_term(pick_d, TRANSFER_REACH_D0)
  place_d = torch.linalg.norm(cube - state.target_pos, dim=-1)
  place = lidbox_reach_term(place_d, TRANSFER_REACH_D0)
  reward = torch.where(
    captured,
    TRANSFER_REWARD_W_CAPTURE + TRANSFER_REWARD_W_PLACE * place,
    TRANSFER_REWARD_W_REACH * reach,
  )
  return reward + TRANSFER_SUCCESS_BONUS * transfer_success(env).float()


def hinsert_reward(env) -> torch.Tensor:
  """Staged shaping for horizontal insertion into the recessed bore.

  Same skeleton as every other mode -- a ``-log(100 * err + 1)`` potential plus
  the SAME sparse 1000 bonus on success -- with two additions the task needs:

  * ``err`` folds the axis misalignment in alongside the position error, exactly
    the way ``insertion_reward`` folds in its quaternion term when
    ``with_rotation`` is set. The weight converts radians to metres-equivalent.
  * a small monotone STAGE bonus (approach -> aligned -> engaged) so the
    align-then-insert ordering is expressed in the reward rather than left to
    the policy to discover. It is 2.0 per stage against a 1000 terminal, so it
    shapes the path without competing with success.

  Position error is measured on the SHAFT TIP against its target on the bore
  axis, so the hidden in-hand offset is inside the error signal.
  """
  from active_perception_arms.robots.so101_constants import (
    HINSERT_R_CLEAR,
    HINSERT_STAGE_BONUS,
    HINSERT_TARGET_DEPTH,
    HINSERT_TOL_ANGLE,
  )
  from active_perception_arms.tasks.insertion.mdp.state import hinsert_errors

  lateral, depth, angle = hinsert_errors(env)
  # Distance from the shaft tip to its target point on the bore axis.
  d_axial = depth - HINSERT_TARGET_DEPTH
  err = torch.sqrt(lateral * lateral + d_axial * d_axial + (HINSERT_ANGLE_WEIGHT * angle) ** 2)
  reward = -torch.log(100.0 * err + 1.0)

  aligned = (angle < HINSERT_TOL_ANGLE) & (lateral < HINSERT_R_CLEAR)
  engaged = aligned & (depth > 0.0)
  stage = aligned.float() + engaged.float()
  reward = reward + HINSERT_STAGE_BONUS * stage

  # The 1000 bonus is paid on the HELD predicate, not the instantaneous one, so
  # its magnitude matches every other mode exactly: those terminate the instant
  # they succeed and collect it once. Paying it per step of the hold would be
  # worth HINSERT_HOLD_STEPS x 1000 and would no longer be "the same magnitude".
  from active_perception_arms.tasks.insertion.mdp.terminations import hinsert_success

  return torch.where(hinsert_success(env), torch.full_like(reward, 1000.0), reward)


# Radians -> metres-equivalent in the folded shaping error. 0.05 m/rad makes the
# 5 deg (0.087 rad) angle tolerance worth 4.4 mm, i.e. comparable to the bore's
# 2 mm lateral clearance -- so neither term can be ignored while the other is
# satisfied.
HINSERT_ANGLE_WEIGHT = 0.05
