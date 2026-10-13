"""Shared world XYZ/pan/tilt target increments for plug and pushing."""

import torch

from .. import plug
from ..config import ARM_JOINTS, CAMERA_HOME, CAMERA_LOOKAT
from .kinematics import BaseFrame, SO101Chain, axis_to_pan_tilt, pan_tilt_to_axis

# camera_action="lookat_v2": absolute camera-position targets inside a box the
# camera arm reaches while viewing the task (north of the optional gap screen),
# plus a look-at point on the work surface. Random actions therefore still give
# task-centred views; targets are rate limited and solved by the same IK.
# The box is the region the per-step IK tracks from the start branch (98% of
# random targets within 3 mm); higher views need another IK branch. Informative
# views of the prongs are all low, and the socket is seen through the gap.
LOOKAT_BOX = {"plug": ((-0.06, 0.18, 0.05), (0.08, 0.30, 0.14))}
LOOKAT_CENTER = {"plug": (0.02, 0.03, 0.03)}
LOOKAT_HALF = {"plug": (0.04, 0.04)}
LOOKAT_SPEED = 0.02  # metres of camera travel per control step
LOOK_POINT_SPEED = 0.02  # metres of look-at travel per control step
# Shared start (and fixed-camera) pose for camera_start="lookat_v2": the single
# camera-arm pose that most often identifies the hidden variant behind the gap
# screen (58% of audited resets; the per-episode best pose reaches 100%; see
# artifacts/diagnosis/views_gap_sampled). Active control can only add by moving.
LOOKAT_START = {"plug": (-0.03, 0.30, 0.08)}

# Calibrated attained poses from the independent reset-view audit, seed130000+.
# They are scene calibration constants, not episode-specific target information.
INSPECTION_POSES = {
    "plug": (0.0577045253, 0.2516049648, 0.0829336207, -2.0527359069, 0.1361678637),
    "push": (-0.2573049432, -0.1877270999, 0.1333050719, 0.6863639783, 0.2598315491),
}


class CameraPoseIK:
    def __init__(self, model, cfg, device="cpu", dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.chain = SO101Chain(
            model,
            "camera_arm/wrist_cam_site",
            tuple("camera_arm/" + j for j in ARM_JOINTS),
            device,
            dtype,
        )
        self.frame = BaseFrame(cfg.camera_base, cfg.camera_base_yaw, device, dtype)
        if cfg.task == "plug":
            home = plug.GIMBAL_HOME
        else:
            axis = self.tensor(CAMERA_LOOKAT) - self.tensor(CAMERA_HOME)
            pan, tilt = axis_to_pan_tilt(axis)
            home = (*CAMERA_HOME, float(pan), float(tilt))
        self.home = self.tensor(home)
        extent = (
            (0.15, 0.15, 0.15, 1.0, 0.9)
            if cfg.camera_workspace == "inspection_v2"
            else plug.GIMBAL_RANGE
        )
        self.low = self.home - self.tensor(extent)
        self.high = self.home + self.tensor(extent)
        self.delta = self.tensor(plug.GIMBAL_DELTA)
        self.initial = (
            self.tensor(INSPECTION_POSES[cfg.task])
            if cfg.camera_start == "inspection_v1"
            else self.home
        )
        self.lookat_enabled = getattr(cfg, "camera_action", "delta_v1") == "lookat_v2"
        if self.lookat_enabled:
            low, high = LOOKAT_BOX[cfg.task]
            self.box_low, self.box_high = self.tensor(low), self.tensor(high)
            self.look_center = self.tensor(LOOKAT_CENTER[cfg.task])
            self.look_half = self.tensor(LOOKAT_HALF[cfg.task])
            self.initial_look = self.look_center[:2].clone()
            start = self.tensor(LOOKAT_START[cfg.task])
            pan, tilt = axis_to_pan_tilt(self.look_center - start)
            self.initial = torch.cat((start, pan[None], tilt[None]))
        if str(device).startswith("cuda"):
            self.solve = torch.compile(self.solve, mode="reduce-overhead", fullgraph=True)

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def solve(self, pose, q):
        return self.chain.solve_ik(
            self.frame.pos_to_base(pose[:, :3]),
            q,
            target_axis=self.frame.vec_to_base(pan_tilt_to_axis(pose[:, 3], pose[:, 4])),
            iters=4,
            damping=0.05,
            max_dq=0.15,
        )

    def lookat(self, position, look, raw, enabled):
        """Rate-limited absolute look-at targets -> (position, look point, gimbal pose)."""
        desired = self.box_low + (raw[:, :3] + 1) * 0.5 * (self.box_high - self.box_low)
        step = (desired - position).clamp(-LOOKAT_SPEED, LOOKAT_SPEED)
        position = position + step * enabled[:, None]
        wanted = self.look_center[:2] + raw[:, 3:5] * self.look_half
        move = (wanted - look).clamp(-LOOK_POINT_SPEED, LOOK_POINT_SPEED)
        look = look + move * enabled[:, None]
        point = torch.cat((look, self.look_center[2].expand(len(look), 1)), dim=-1)
        pan, tilt = axis_to_pan_tilt(point - position)
        return position, look, torch.cat((position, pan[:, None], tilt[:, None]), dim=-1)
