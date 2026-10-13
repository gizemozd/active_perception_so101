"""Seated-insertion rules for ``hidden_prongs_v2``, shared by native and Warp backends.

Every function takes batched torch tensors of world-local positions (environment
origin already subtracted). Native diagnostics and MjLab/Warp training call the
same code, so success, failure and reward cannot drift apart between backends.

Why v2 exists (see docs/DIAGNOSIS_2026-10-13.md):

* v1 success is a 2 mm sphere centred 2 mm *above* the seated pose. The socket
  plate top is at z = 14 mm and the goal is z = 16 mm, so a plug pushed fully
  home (housing on the plate, both prongs inside the holes) measures 2.0-3.0 mm
  and never counts. v2 scores the physical state: prongs engaged >= 8 mm and the
  body inside the hole clearance.
* v1 lets a wrist-only policy "probe" the socket blindly once the horizon is
  long enough. v2 optionally terminates a misaligned press on the plate surface
  (a jammed connector), so the hidden prong offset must be perceived, not
  found by repeated contact.
* v1's ported reward pays +1000 on every step inside the sphere, which RSL-RL's
  clipped value loss (0.2 per update, no value normalisation) cannot track. The
  v2 reward is bounded: staged progress differences, +10 on held success and a
  -5 jam penalty.
"""

import torch

from . import plug

PLATE_TOP = 0.014  # socket plate spans z in [0, 14] mm
PLATE_HALF = (0.05, 0.035)
PRONG_LENGTH = 2 * plug.PRONG_HALF[2]  # 12 mm below the housing
# Holes leave 2 mm clearance per side; soft contacts add a few tenths of a mm.
SEAT_XY_TOLERANCE = 0.0025
# Body bottom at most 2 mm above the target depth, i.e. prongs engaged >= 8 mm.
# The housing physically rests on the plate 2 mm *below* the v1 goal height.
SEAT_DEPTH_TOLERANCE = 0.002
# Height above the target depth at which prong tips clear the plate top by 4 mm
# (14 mm; the original spawn is 19 mm above the target depth).
SAFE_HEIGHT = PLATE_TOP + 0.004 + PRONG_LENGTH - plug.INSERT_Z
# Descent credit starts above the highest reachable housing height (29 mm).
DESCENT_TOP = 0.030
# A press counts as jammed when the tips reach within 1 mm of the plate top
# while a prong lies entirely off its hole (hole half-width 6 mm), i.e. it is
# pressed onto the solid plate. Wrong-variant goals are 15-30 mm away, so blind
# probing always jams; wiggling within a hole's footprint does not. (A 3 mm
# draft threshold terminated 84% of random explorations from aligned starts.)
JAM_TIP_MARGIN = 0.001
JAM_XY = plug.PRONG_HALF[0] + 0.002
# Training-only reset curriculum (plug_assist_fraction): a fraction of resets
# start already aligned above the goal, so the descent/seat skill is practised
# before the policy can reach that state itself. Evaluation keeps the original
# reset distribution. Heights are the housing bottom above the target depth.
ASSIST_XY = 0.0015
ASSIST_HEIGHT = (0.004, 0.019)
SUCCESS_BONUS = 10.0
JAM_PENALTY = 5.0
SHAPING_SCALE = 2.5

# Optional optical "gap screen" (plug_screen="gap_v1"): a collision-free wall
# between the north camera arm and the task, with one gap whose lateral position
# is resampled every episode. Low views that reveal the prongs exist only through
# the gap, so no single fixed placement sees them in every episode, while the
# camera arm can realign itself. High views over the wall still see the socket.
SCREEN_Y = 0.145
SCREEN_HALF_THICKNESS = 0.003
SCREEN_HALF_HEIGHT = 0.06
SCREEN_SEGMENT_HALF_LENGTH = 0.25
SCREEN_GAP = 0.035
SCREEN_GAP_RANGE = (-0.06, 0.10)
# The gap is drawn uniformly from the centres for which a camera in the low band
# of the look-at box (y = 0.21 m, x in this range) can see the plug's underside
# through it at reset. A gap-aligned view is then always reachable, while any
# single fixed pose sees through the gap only in a fraction of episodes.
SCREEN_VIEW_Y = 0.21
SCREEN_VIEW_X = (-0.055, 0.075)


def sample_gap(u, plug_xy):
    """Gap centre (world-local x) for uniform draws u in [0, 1] and the reset plug XY."""
    px, py = plug_xy[:, 0], plug_xy[:, 1]
    scale = (SCREEN_VIEW_Y - py) / (SCREEN_Y - py)
    low = (px + (SCREEN_VIEW_X[0] - px) / scale).clamp(*SCREEN_GAP_RANGE)
    high = (px + (SCREEN_VIEW_X[1] - px) / scale).clamp(*SCREEN_GAP_RANGE)
    return low + u * (high - low)


def screen_segment_centers(gap_x):
    """World-local XYZ centres of the left and right wall segments."""
    offset = SCREEN_GAP / 2 + SCREEN_SEGMENT_HALF_LENGTH
    z = SCREEN_HALF_HEIGHT
    left = torch.stack(
        (gap_x - offset, torch.full_like(gap_x, SCREEN_Y), torch.full_like(gap_x, z)), -1
    )
    right = torch.stack(
        (gap_x + offset, torch.full_like(gap_x, SCREEN_Y), torch.full_like(gap_x, z)), -1
    )
    return left, right


def errors(obj, goal):
    """Lateral error and height of the housing bottom above the target depth."""
    return (obj[:, :2] - goal[:, :2]).norm(dim=-1), obj[:, 2] - goal[:, 2]


def seated(obj, goal):
    """Prongs inside both holes and engaged at least 8 mm (one-sided depth test)."""
    xy, height = errors(obj, goal)
    return (xy < SEAT_XY_TOLERANCE) & (height < SEAT_DEPTH_TOLERANCE)


def prong_centers(obj, offset):
    """World-local XY of both prongs; the grasp keeps identity plug orientation."""
    center = obj[:, :2] + offset
    spacing = torch.zeros_like(center)
    spacing[:, 0] = plug.HALF_SPACING
    return torch.stack((center + spacing, center - spacing), dim=1)


def over_plate(obj, offset, socket):
    """Whether either prong footprint overlaps the plate's top face."""
    half = torch.tensor(PLATE_HALF, device=obj.device, dtype=obj.dtype) + plug.PRONG_HALF[0]
    rel = (prong_centers(obj, offset) - socket[:, None, :2]).abs()
    return ((rel[..., 0] < half[0]) & (rel[..., 1] < half[1])).any(dim=-1)


def jammed(obj, goal, offset, socket):
    """Misaligned prongs pressed onto the plate surface (would damage a connector)."""
    xy, _ = errors(obj, goal)
    tips = obj[:, 2] - PRONG_LENGTH
    low = tips < PLATE_TOP + JAM_TIP_MARGIN
    return low & (xy > JAM_XY) & over_plate(obj, offset, socket)


def potential(obj, goal):
    """Staged lift/align/descend potential in [-2, 4]; maximal when seated.

    Once aligned, descent is rewarded over the whole reachable height range
    (an earlier draft rewarded it only below SAFE_HEIGHT and left a zero-gradient
    band above it, where privileged PPO stalled). Being low while misaligned is
    penalised, so the shortest path through the plate (the v1 "XM jam") is not.
    """
    xy, height = errors(obj, goal)
    coarse = torch.exp(-xy / 0.02)
    fine = torch.exp(-xy / 0.003)
    descent = ((DESCENT_TOP - height) / (DESCENT_TOP + 0.002)).clamp(0, 1)
    danger = ((SAFE_HEIGHT - height) / (SAFE_HEIGHT + 0.002)).clamp(0, 1)
    off_hole = 1 - torch.exp(-xy / JAM_XY)
    return coarse + fine + 2 * fine * descent - 2 * off_hole * danger


def reward(phi, previous_phi, valid, success, jam):
    """Difference shaping (telescopes to the final potential) plus terminal events."""
    progress = torch.where(valid, phi - previous_phi, torch.zeros_like(phi))
    return SHAPING_SCALE * progress + SUCCESS_BONUS * success.float() - JAM_PENALTY * jam.float()


def _segment_hits_box(start, end, low, high):
    """Slab test: does the open segment start->end cross the axis-aligned box?"""
    direction = end - start
    safe = torch.where(direction.abs() < 1e-9, torch.full_like(direction, 1e-9), direction)
    t1, t2 = (low - start) / safe, (high - start) / safe
    near = torch.minimum(t1, t2).amax(-1)
    far = torch.maximum(t1, t2).amin(-1)
    return (near <= far) & (far > 1e-6) & (near < 1 - 1e-6)


def prong_visibility(camera, axis, obj, offset, gap=None, half_fov=0.4229):
    """Fraction of the two prong tips with an unobstructed line of sight.

    Analytic, training-only proxy for the camera-arm view: frustum cone (half the
    vertical FOV), occlusion by the plug housing and, when present, by the two gap
    screen segments. Arms and the collar are ignored. World-local coordinates.
    """
    tips = torch.cat(
        (
            prong_centers(obj, offset),
            (obj[:, 2] - PRONG_LENGTH + 0.001)[:, None, None].expand(-1, 2, 1),
        ),
        dim=-1,
    )
    cam = camera[:, None].expand_as(tips)
    ray = tips - cam
    cosine = (ray * axis[:, None]).sum(-1) / ray.norm(dim=-1).clamp_min(1e-9)
    visible = cosine > torch.cos(torch.tensor(half_fov, dtype=obj.dtype, device=obj.device))
    half = torch.tensor(plug.BODY_HALF, dtype=obj.dtype, device=obj.device)
    low = obj.clone()
    low[:, :2] -= half[:2]
    high = obj.clone()
    high[:, :2] += half[:2]
    high[:, 2] += 2 * half[2]
    visible &= ~_segment_hits_box(cam, tips, low[:, None], high[:, None])
    if gap is not None:
        size = torch.tensor(
            (SCREEN_SEGMENT_HALF_LENGTH, SCREEN_HALF_THICKNESS, SCREEN_HALF_HEIGHT),
            dtype=obj.dtype,
            device=obj.device,
        )
        for center in screen_segment_centers(gap.to(obj.dtype)):
            visible &= ~_segment_hits_box(
                cam, tips, (center - size)[:, None], (center + size)[:, None]
            )
    return visible.float().mean(-1)
