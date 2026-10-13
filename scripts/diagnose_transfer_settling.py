#!/usr/bin/env python3
"""Hold a captured failed placement at rest to distinguish contact chatter from control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from active_perception_arms import mdp
from active_perception_arms.config import JOINTS, Experiment
from active_perception_arms.environment import make_env_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    trace = np.load(args.trace)
    records = []
    for name in ("current", "iterations100", "timestep1ms", "rolling5x"):
        cfg = Experiment(
            task="transfer",
            condition="wrist_static",
            occlusion="clean",
            render_sensors=False,
            randomize=False,
            num_envs=8,
            timestep=0.001 if name == "timestep1ms" else 0.002,
            decimation=40 if name == "timestep1ms" else 20,
        )
        setup = make_env_cfg(cfg)
        setup.terminations = {}
        if name == "iterations100":
            setup.sim.mujoco.iterations = 100
        if name == "rolling5x":
            original = setup.scene.spec_fn

            def changed(spec, original=original):
                original(spec)
                spec.geom("object/block").friction = [0.8, 0.005, 0.0005]

            setup.scene.spec_fn = changed
        env = ManagerBasedRlEnv(setup, device="cuda:0")
        try:
            env.reset()
            s = mdp.state(env)
            env.sim.data.qpos[:] = torch.as_tensor(trace["qpos"][-1], device="cuda")
            env.sim.data.qpos[:, s.object_qpos_address : s.object_qpos_address + 3] += (
                env.scene.env_origins - torch.as_tensor(trace["origins"][-1], device="cuda")
            )
            env.sim.data.qvel[:] = 0
            for body, key in [("fixture/fixture", "fixture"), ("occluder/panel", "panel")]:
                index = env.sim.mj_model.body(body).mocapid[0]
                env.sim.data.mocap_pos[:, index] = torch.as_tensor(trace[key][-1], device="cuda")
            action = torch.zeros(8, cfg.action_dim, device="cuda")
            term = env.action_manager.get_term("arms")
            for arm in ("manipulator", "camera_arm"):
                addresses = [int(env.sim.mj_model.joint(arm + "/" + j).qposadr[0]) for j in JOINTS]
                term.targets[arm][:] = env.sim.data.qpos[:, addresses]
            action[:, 5] = term.targets["manipulator"][:, 5] / 0.6 - 1
            env.sim.forward()
            samples = []
            for step in range(150):
                env.step(action)
                qvel = env.sim.data.qvel[:, s.object_dof_address : s.object_dof_address + 6]
                if step >= 100:
                    samples.append(
                        torch.stack((qvel[:, :3].norm(dim=-1), qvel[:, 3:].norm(dim=-1)), dim=-1)
                        .cpu()
                        .numpy()
                    )
            values = np.stack(samples)
            record = dict(
                case=name,
                linear_max=float(values[:, :, 0].max()),
                angular_max=float(values[:, :, 1].max()),
                angular_mean=float(values[:, :, 1].mean()),
                settled_fraction=float(
                    ((values[:, :, 0] < 0.035) & (values[:, :, 1] < 0.5)).mean()
                ),
            )
            records.append(record)
            args.output.write_text(
                json.dumps(dict(trace=str(args.trace), cases=records), indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
