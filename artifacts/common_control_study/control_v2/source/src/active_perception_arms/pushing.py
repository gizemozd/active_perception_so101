"""Current-state, contact-free planar-pushing completion checks."""

import math
from dataclasses import replace

import numpy as np
import torch

from . import transfer
from .robots.kinematics import quat_wxyz_to_mat

ARC_HALF_ANGLE = math.radians(80)
ARC_RADIUS_MIN = 0.050
ARC_RADIUS_MAX = 0.120
ARC_V1_RADIUS_MAX = 0.070


def arc_offset(unit_samples, layout="arc_v2"):
    """Uniform angle [-80,+80] deg from +Y and independent radius [50,120] mm.

    Supports NumPy native resets and batched Torch/Warp resets. This samples
    radius uniformly, not area uniformly. A pair of 0.5 values is the nominal reset.
    """
    angle = (2 * unit_samples[..., 0] - 1) * ARC_HALF_ANGLE
    radius_max = ARC_V1_RADIUS_MAX if layout == "arc_v1" else ARC_RADIUS_MAX
    radius = ARC_RADIUS_MIN + unit_samples[..., 1] * (radius_max - ARC_RADIUS_MIN)
    if isinstance(unit_samples, torch.Tensor):
        return torch.stack((radius * angle.sin(), radius * angle.cos()), dim=-1)
    return np.stack((radius * np.sin(angle), radius * np.cos(angle)), axis=-1)


def tool_offset(model, device):
    site = model.site("manipulator/gripperframe")
    rotation = quat_wxyz_to_mat(
        torch.as_tensor(site.quat, device=device, dtype=torch.float32)[None]
    )[0]
    offset = torch.as_tensor(
        model.geom("pushing_tip").pos - site.pos, device=device, dtype=torch.float32
    )
    return rotation.T @ offset


def tool_position(env, state):
    tcp, rotation = state.current_tcp_fk(env.sim.data.qpos[:, state.manip_qpos_addresses])
    return tcp + state.manip_base + env.scene.env_origins + (rotation @ state.push_tool_offset)


def contact_sensors():
    return [replace(transfer.contact_sensors()[2], name="push_manipulator")]


def warp_clear(env):
    data = env.scene["push_manipulator"].data
    touching = ((data.found > 0) & (data.force.norm(dim=-1) >= transfer.CONTACT_FORCE_MIN)).any(-1)
    return ~touching


def native_clear(env):
    return not transfer.native_contacts(env)["manipulator"]
