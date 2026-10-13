#!/usr/bin/env python3
"""Reproduce one hard XM approach and inspect IK/contact without changing training."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from active_perception_arms import plug
from active_perception_arms.config import Experiment
from active_perception_arms.native import NativeEnv, ScriptedPolicy


def run(lift_first):
    env = NativeEnv(Experiment(plug_variant="xm", condition="wrist_static", render_sensors=False))
    env.reset(10011)
    controller = ScriptedPolicy(env)
    start = env.tcp_target.copy()
    held, successful, closest = 0, False, float("inf")
    for step in range(88):
        action = controller()
        if lift_first and step < 33:
            raised = start.copy()
            raised[2] = 0.09
            action[:3] = (
                2 * (raised - plug.ACTION_LOW) / (np.array(plug.ACTION_HIGH) - plug.ACTION_LOW) - 1
            )
        won = env.step(action)
        held = held + 1 if won else 0
        successful |= held >= 3
        closest = min(closest, float(np.linalg.norm(env.object_pos - env.goal)))
        if successful:
            break
    ik = env.plug_ik
    requested = ik.tensor(env.tcp_target)[None]
    q = ik.tensor(env.targets["manipulator"][:5])[None]
    warm_pos, _ = ik.manip.fk(q)
    cold_pos, _ = ik.manip.fk(ik.reset_manip(requested))
    contacts = []
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        force = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, index, force)
        names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in contact.geom]
        if abs(force[0]) > 0.1:
            contacts.append({"geoms": names, "normal_force_N": float(force[0])})
    return {
        "seed": 10011,
        "variant": "xm",
        "lift_first": lift_first,
        "held_success": successful,
        "steps": step + 1,
        "closest_error_m": closest,
        "final_body_error_m": float(np.linalg.norm(env.object_pos - env.goal)),
        "tcp_target_m": env.tcp_target.tolist(),
        "physical_tcp_m": env.tcp.tolist(),
        "body_goal_m": env.goal.tolist(),
        "warm_ik_position_residual_m": float(
            (ik.manip_frame.pos_to_world(warm_pos) - requested).norm()
        ),
        "cold_ik_position_residual_m": float(
            (ik.manip_frame.pos_to_world(cold_pos) - requested).norm()
        ),
        "contacts": sorted(contacts, key=lambda row: -row["normal_force_N"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {
        "backend": "native MuJoCo",
        "learner_updates": 0,
        "interpretation": "One known hard case; not aggregate learned-policy performance.",
        "cases": [run(False), run(True)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
