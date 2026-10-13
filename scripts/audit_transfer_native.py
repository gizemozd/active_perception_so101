#!/usr/bin/env python3
"""Measure transfer grasp, lift, placement and release on real native physics."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from active_perception_arms import transfer
from active_perception_arms.config import Experiment
from active_perception_arms.native import NativeEnv, ScriptedPolicy


def contacts(env):
    result = dict(fixed=False, moving=False, manipulator=False, floor=False)
    forces = []
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        names = [env.model.geom(int(g)).name for g in contact.geom]
        if "object/block" not in names:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, index, force)
        if abs(force[0]) < 0.001:
            continue
        other = names[1] if names[0] == "object/block" else names[0]
        result["fixed"] |= other.startswith("manipulator/fixed_jaw")
        result["moving"] |= other.startswith("manipulator/moving_jaw")
        result["manipulator"] |= other.startswith("manipulator/")
        result["floor"] |= other == "fixture/floor"
        forces.append([other, float(force[0])])
    return result, forces


def episode(env, seed):
    env.reset(seed)
    controller = ScriptedPolicy(env)
    rows, held, won, physical_hold = [], 0, False, 0
    ever_lifted = False
    dof = int(env.model.joint("object/free").dofadr[0])
    for step in range(round(env.cfg.episode_seconds / env.cfg.step_dt)):
        action = controller()
        success = env.step(action)
        contact, forces = contacts(env)
        obj, goal, tcp = env.object_pos, env.goal, env.tcp
        bilateral = contact["fixed"] and contact["moving"]
        ever_lifted |= bilateral and obj[2] >= 0.081
        physical = success and ever_lifted and contact["floor"] and not contact["manipulator"]
        physical_hold = physical_hold + 1 if physical else 0
        held = held + 1 if success else 0
        won |= held >= (transfer.HOLD_STEPS if transfer.is_v2(env.cfg) else 3)
        reach = float(np.exp(-np.linalg.norm(tcp - obj) / 0.05))
        lift = float(np.clip((obj[2] - 0.056) / 0.05, 0, 1))
        place = float(np.exp(-np.linalg.norm(obj - goal) / 0.045))
        rows.append(
            dict(
                step=step + 1,
                stage=controller.stage,
                object=obj.tolist(),
                goal=goal.tolist(),
                tcp=tcp.tolist(),
                contacts=contact,
                speed=float(np.linalg.norm(env.data.qvel[dof : dof + 3])),
                angular_speed=float(np.linalg.norm(env.data.qvel[dof + 3 : dof + 6])),
                gripper=float(env.data.qpos[env.qadr["manipulator"][5]]),
                baseline_potential=0.3 * reach + 0.3 * lift + 1.4 * place,
                instantaneous_success=bool(success),
                contact_forces=forces,
            )
        )
        if physical_hold >= 5:
            break
    return dict(
        seed=seed,
        randomized=env.cfg.randomize,
        success=won,
        physical_released_success=physical_hold >= 5,
        lifted_with_both_jaws=bool(ever_lifted),
        max_height=max(row["object"][2] for row in rows),
        final_error=float(np.linalg.norm(env.object_pos - env.goal)),
        final_stage=controller.stage,
        steps=len(rows),
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = []
    for randomize in (False, True):
        cfg = Experiment(
            task="transfer",
            condition="wrist_static",
            occlusion="clean",
            num_envs=1,
            render_sensors=False,
            randomize=randomize,
        )
        env = NativeEnv(cfg)
        for seed in range(10000, 10000 + args.episodes) if randomize else [0]:
            record = episode(env, seed)
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "rows"}), flush=True)
    report = dict(
        backend="native MuJoCo",
        task_revision=cfg.task_revision,
        reward_profile=cfg.reward_profile,
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *Path("src/active_perception_arms").glob("*.py")]
        },
        training_updates=0,
        records=records,
        summary={
            "episodes": len(records),
            "successes": sum(r["success"] for r in records),
            "physical_released_successes": sum(r["physical_released_success"] for r in records),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
