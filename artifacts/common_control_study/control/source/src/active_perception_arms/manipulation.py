"""Opt-in Cartesian controls and dense shaping for transfer/push learning studies.

The controller uses only commanded TCP and robot joints. Object/goal state is
used exclusively by rewards, the critic, and privileged feasibility diagnostics.
"""

import numpy as np
import torch

from .config import ARM_JOINTS, MANIP_BASE
from .robots.kinematics import SO101Chain, grasp_site_rotation


def bounds(task, revision="legacy_v1"):
    if task == "transfer":
        return (-0.095, -0.09, 0.048), (0.075, 0.095, 0.16)
    if revision == "wide_v2":
        return (-0.17, -0.14, 0.037), (0.12, 0.13, 0.13)
    return (-0.10, -0.13, 0.037), (0.065, 0.11, 0.13)


def max_delta(task):
    return 0.003 if task == "transfer" else 0.0015


class CartesianIK:
    def __init__(self, model, task, device="cpu", dtype=torch.float32, revision="legacy_v1"):
        self.chain = SO101Chain(
            model,
            "manipulator/gripperframe",
            tuple("manipulator/" + j for j in ARM_JOINTS),
            device,
            dtype,
        )
        self.base = torch.tensor(MANIP_BASE, device=device, dtype=dtype)
        self.rotation = torch.as_tensor(
            grasp_site_rotation(model, prefix="manipulator/", yaw=-np.pi / 2),
            device=device,
            dtype=dtype,
        )[None]
        self.low, self.high = [
            torch.tensor(v, device=device, dtype=dtype) for v in bounds(task, revision)
        ]
        self.delta = max_delta(task)
        if str(device).startswith("cuda"):
            self.solve = torch.compile(self.solve, fullgraph=True)

    def solve(self, tcp, q):
        return self.chain.solve_ik(
            tcp - self.base,
            q,
            target_rot=self.rotation.expand(len(q), -1, -1),
            iters=6,
            damping=0.006,
            max_dq=0.15,
        )


def transfer_density(f, tcp, obj):
    """Dense stage reward, with aligned approach and physical grasp/lift credit."""
    grasp = obj + obj.new_tensor([0, -0.008, 0.003])
    reach = torch.exp(-(tcp - grasp).norm(dim=-1) / 0.04)
    grasp_lift = 1 + f["lift"]
    transport = 2 + torch.exp(-f["distance"] / 0.07)
    phi = torch.where(f["bilateral"], grasp_lift, reach)
    # Pickup history alone cannot earn transport credit for a dropped cube.
    phi = torch.where(f["lifted"] & (f["bilateral"] | f["placed"]), transport, phi)
    withdrawal = ((f["hand_distance"] - 0.012) / (0.045 - 0.012)).clamp(0, 1)
    return torch.where(f["lifted"] & f["released"], 3 + withdrawal, phi)


def transfer_transport_density(f, tcp, obj):
    """Amplify the lifted-object distance gradient after pickup stalls at source."""
    grasp = obj + obj.new_tensor([0, -0.008, 0.003])
    reach = torch.exp(-(tcp - grasp).norm(dim=-1) / 0.04)
    grasp_lift = 1 + f["lift"]
    # At the nominal 12 cm transfer distance this yields a useful gradient per
    # 3 mm Cartesian step; the previous unit-amplitude profile rewarded holding
    # at the source more than the small transport improvement.
    transport = 2 + 6 * torch.exp(-f["distance"] / 0.06)
    phi = torch.where(f["bilateral"], grasp_lift, reach)
    phi = torch.where(f["lifted"] & (f["bilateral"] | f["placed"]), transport, phi)
    withdrawal = ((f["hand_distance"] - 0.012) / (0.045 - 0.012)).clamp(0, 1)
    return torch.where(f["lifted"] & f["released"], 3 + withdrawal, phi)


def push_density(tcp, obj, goal, success_tolerance=0.015):
    delta = goal[:, :2] - obj[:, :2]
    distance = delta.norm(dim=-1)
    behind = obj.clone()
    behind[:, :2] -= 0.025 * delta / distance[:, None].clamp_min(1e-6)
    behind[:, 2] += 0.031
    reach = torch.exp(-(tcp - behind).norm(dim=-1) / 0.045)
    place = torch.exp(-distance / 0.045)
    tool = tcp + tcp.new_tensor([0, 0, -0.031])
    withdraw = ((tool - obj).norm(dim=-1) / 0.065).clamp(0, 1)
    return 0.5 * reach + 2 * place + (distance < success_tolerance) * withdraw
