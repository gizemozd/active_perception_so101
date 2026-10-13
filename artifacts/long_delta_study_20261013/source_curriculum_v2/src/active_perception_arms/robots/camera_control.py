"""Shared world XYZ/pan/tilt target increments for plug and pushing."""

import torch

from .. import plug
from ..config import ARM_JOINTS, CAMERA_HOME, CAMERA_LOOKAT
from .kinematics import BaseFrame, SO101Chain, axis_to_pan_tilt, pan_tilt_to_axis

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
