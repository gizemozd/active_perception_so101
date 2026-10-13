"""Physical pickup/place/release semantics and staged transfer potential."""

import mujoco
import numpy as np
import torch

REVISION = "pick_place_v2"
SOURCE_CENTER_Z = 0.056
LIFT_HEIGHT = 0.025
XY_TOLERANCE = 0.018
Z_TOLERANCE = 0.006
WITHDRAW_DISTANCE = 0.045
LINEAR_SPEED_LIMIT = 0.035
ANGULAR_SPEED_LIMIT = 0.5
CONTACT_FORCE_MIN = 0.001
HOLD_STEPS = 5
DISCOUNT = 0.99
SUCCESS_BONUS = 25.0


def is_v2(cfg):
    return cfg.task == "transfer" and getattr(cfg, "task_revision", None) == REVISION


def features(tcp, obj, goal, linear_speed, angular_speed, contacts, lifted, origin_z=0.0):
    """Batched tensors; contacts are privileged reward/critic inputs only."""
    distance = torch.linalg.vector_norm(obj - goal, dim=-1)
    hand_distance = torch.linalg.vector_norm(tcp - obj, dim=-1)
    bilateral = contacts["fixed"] & contacts["moving"]
    lift = ((obj[:, 2] - origin_z - SOURCE_CENTER_Z) / LIFT_HEIGHT).clamp(0, 1)
    lifted = lifted | (bilateral & (lift >= 1))
    placed = (
        (torch.linalg.vector_norm(obj[:, :2] - goal[:, :2], dim=-1) < XY_TOLERANCE)
        & ((obj[:, 2] - goal[:, 2]).abs() < Z_TOLERANCE)
        & contacts["floor"]
    )
    released = placed & ~contacts["manipulator"]
    settled = (linear_speed < LINEAR_SPEED_LIMIT) & (angular_speed < ANGULAR_SPEED_LIMIT)
    success = lifted & released & settled & (hand_distance > WITHDRAW_DISTANCE)
    return dict(
        distance=distance,
        hand_distance=hand_distance,
        bilateral=bilateral,
        lift=lift,
        lifted=lifted,
        placed=placed,
        released=released,
        settled=settled,
        instantaneous_success=success,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
    )


def potential(f):
    """Reach [0,1], grasp/lift [1,2], transport [2,3], release/withdraw [3,4]."""
    reach = torch.exp(-f["hand_distance"] / 0.04)
    phi = torch.where(f["bilateral"], 1 + f["lift"], reach)
    phi = torch.where(f["lifted"], 2 + torch.exp(-f["distance"] / 0.07), phi)
    withdrawal = ((f["hand_distance"] - 0.012) / (WITHDRAW_DISTANCE - 0.012)).clamp(0, 1)
    return torch.where(f["lifted"] & f["released"], 3 + withdrawal, phi)


def native_contacts(env):
    """Read actual load-bearing contacts; never attach or reposition the cube."""
    result = dict(fixed=False, moving=False, manipulator=False, floor=False)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        names = [env.model.geom(int(g)).name for g in contact.geom]
        if "object/block" not in names:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, index, force)
        if abs(force[0]) < CONTACT_FORCE_MIN:
            continue
        other = names[1] if names[0] == "object/block" else names[0]
        result["fixed"] |= other.startswith("manipulator/fixed_jaw")
        result["moving"] |= other.startswith("manipulator/moving_jaw")
        result["manipulator"] |= other.startswith("manipulator/") or other == "pushing_tip"
        result["floor"] |= other == "fixture/floor"
    return result


def native_features(env):
    address = int(env.model.joint("object/free").dofadr[0])
    f = features(
        torch.as_tensor(env.tcp[None]),
        torch.as_tensor(env.object_pos[None]),
        torch.as_tensor(env.goal[None]),
        torch.tensor([np.linalg.norm(env.data.qvel[address : address + 3])]),
        torch.tensor([np.linalg.norm(env.data.qvel[address + 3 : address + 6])]),
        {key: torch.tensor([value]) for key, value in native_contacts(env).items()},
        torch.tensor([env.transfer_lifted]),
    )
    env.transfer_lifted = f["lifted"].item()
    return f


def contact_sensors():
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    sensors = []
    for name, mode, pattern, entity in (
        ("fixed", "geom", "fixed_jaw.*", "manipulator"),
        ("moving", "geom", "moving_jaw.*", "manipulator"),
        ("manipulator", "subtree", "base", "manipulator"),
        ("floor", "geom", "floor", "fixture"),
    ):
        sensors.append(
            ContactSensorCfg(
                name="transfer_" + name,
                primary=ContactMatch(mode=mode, pattern=pattern, entity=entity),
                secondary=ContactMatch(mode="geom", pattern="block", entity="object"),
                secondary_policy="error",
                fields=("found", "force"),
                reduce="maxforce",
            )
        )
    return sensors


def warp_features(env, state, tcp, obj, goal, *, update=True):
    contacts = {}
    for name in ("fixed", "moving", "manipulator", "floor"):
        data = env.scene["transfer_" + name].data
        contacts[name] = ((data.found > 0) & (data.force.norm(dim=-1) >= CONTACT_FORCE_MIN)).any(
            dim=-1
        )
    address = state.object_dof_address
    velocity = env.sim.data.qvel[:, address : address + 6]
    f = features(
        tcp,
        obj,
        goal,
        velocity[:, :3].norm(dim=-1),
        velocity[:, 3:].norm(dim=-1),
        contacts,
        state.transfer_lifted,
        env.scene.env_origins[:, 2],
    )
    if update:
        state.transfer_lifted.copy_(f["lifted"])
    return f
