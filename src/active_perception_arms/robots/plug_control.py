"""Batched fixed-orientation TCP and five-axis camera control for the legacy plug."""

import numpy as np
import torch

from .. import plug
from ..config import ARM_JOINTS, MANIP_BASE
from .kinematics import BaseFrame, SO101Chain, grasp_site_rotation, pan_tilt_to_axis


class PlugIK:
    def __init__(self, model, device="cpu", dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.manip = SO101Chain(
            model,
            "manipulator/gripperframe",
            tuple("manipulator/" + j for j in ARM_JOINTS),
            device,
            dtype,
        )
        self.camera = SO101Chain(
            model,
            "camera_arm/wrist_cam_site",
            tuple("camera_arm/" + j for j in ARM_JOINTS),
            device,
            dtype,
        )
        self.manip_frame = BaseFrame(MANIP_BASE, 0, device, dtype)
        self.camera_frame = BaseFrame(plug.CAMERA_BASE, plug.CAMERA_BASE_YAW, device, dtype)
        self.rotation = self.tensor(
            grasp_site_rotation(model, prefix="manipulator/", yaw=-np.pi / 2)
        )
        self.low, self.high = self.tensor(plug.ACTION_LOW), self.tensor(plug.ACTION_HIGH)
        self.home = self.tensor(plug.GIMBAL_HOME)
        self.gimbal_low = self.home - self.tensor(plug.GIMBAL_RANGE)
        self.gimbal_high = self.home + self.tensor(plug.GIMBAL_RANGE)
        self.gimbal_delta = self.tensor(plug.GIMBAL_DELTA)
        # Fixed iteration counts and tensor-only math: no CPU sync or per-physics-
        # substep Jacobian calls. CUDA compiles the small DLS kernels together.
        if str(device).startswith("cuda"):
            self.manip_step = torch.compile(self.manip_step, mode="reduce-overhead", fullgraph=True)
            self.camera_step = torch.compile(
                self.camera_step, mode="reduce-overhead", fullgraph=True
            )

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def reset_manip(self, tcp):
        pos = self.manip_frame.pos_to_base(tcp)
        rot = self.rotation.expand(len(tcp), 3, 3)
        return self.manip.solve_ik_staged(
            pos, rot, site_axis=(1, 0, 0), iters_axis=80, iters_full=50, damping=0.008
        )

    def manip_step(self, tcp, previous_q):
        pos = self.manip_frame.pos_to_base(tcp)
        rot = self.rotation.expand(len(tcp), 3, 3)
        q = self.manip.solve_ik(
            pos,
            previous_q,
            target_axis=rot[:, :, 0],
            site_axis=(1, 0, 0),
            iters=8,
            damping=0.01,
            max_dq=0.3,
        )
        return self.manip.solve_ik(pos, q, target_rot=rot, iters=6, damping=0.01, max_dq=0.3)

    def camera_step(self, gimbal, previous_q):
        return self.camera.solve_ik(
            self.camera_frame.pos_to_base(gimbal[:, :3]),
            previous_q,
            target_axis=self.camera_frame.vec_to_base(pan_tilt_to_axis(gimbal[:, 3], gimbal[:, 4])),
            iters=4,
            damping=0.05,
            max_dq=0.15,
        )

    def camera_reset(self, gimbal, seed):
        return self.camera.solve_ik(
            self.camera_frame.pos_to_base(gimbal[:, :3]),
            seed,
            target_axis=self.camera_frame.vec_to_base(pan_tilt_to_axis(gimbal[:, 3], gimbal[:, 4])),
            iters=120,
            damping=0.01,
            max_dq=0.3,
        )
