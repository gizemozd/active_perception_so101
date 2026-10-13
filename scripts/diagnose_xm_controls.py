#!/usr/bin/env python3
"""Paired frozen-policy control interventions for XM failures; no learner updates."""

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from active_perception_arms import mdp, plug
from active_perception_arms.config import saved_experiment
from active_perception_arms.environment import make_env
from active_perception_arms.evaluate import load_actor
from active_perception_arms.evaluation_audit import TerminationAudit


class ControlAudit(TerminationAudit):
    def compute(self):
        done = super().compute()
        env, arms, state = self.env, self.env.action_manager.get_term("arms"), mdp.state(self.env)
        _, _, _, goal = mdp.positions(env)
        obj = env.sim.data.qpos[:, self.object_qpos_address : self.object_qpos_address + 3]
        physical_tcp, _ = arms.plug_ik.manip.fk(env.sim.data.qpos[:, self.qaddresses])
        physical_tcp = arms.plug_ik.manip_frame.pos_to_world(physical_tcp)
        self.snapshot.update(
            body=obj.clone(),
            goal=goal.clone(),
            target=arms.tcp_target.clone(),
            physical_tcp=physical_tcp.clone(),
            object_velocity=env.sim.data.qvel[
                :, state.object_dof_address : state.object_dof_address + 6
            ].clone(),
            orientation=mdp.plug_orientation_error(env).clone(),
            target_yaw=arms.wrist_yaw_target.clone(),
        )
        return done

    def __init__(self, env):
        super().__init__(env)
        self.qaddresses = [
            int(env.sim.mj_model.joint(name).qposadr[0])
            for name in env.action_manager.get_term("arms").plug_ik.manip.joint_names
        ]


def analyze(cfg, checkpoint, mode, seeds, output):
    env = make_env(cfg, "cuda:0")
    audit = ControlAudit(env)
    records, traces, initial_hashes = [], [], []
    try:
        obs, _ = env.reset(seed=seeds[0])
        actor = load_actor(checkpoint, obs, "cuda:0")
        term = env.action_manager.get_term("arms")
        low, high = term.plug_ik.low, term.plug_ik.high
        with torch.inference_mode():
            for seed in seeds:
                obs, _ = env.reset(seed=seed)
                audit.reset_batch()
                actor.reset()
                initial_hashes.extend(audit.initial_state_hashes(obs, cfg.sensors))
                finished = torch.zeros(cfg.num_envs, dtype=torch.bool, device="cuda")
                closest = torch.full((cfg.num_envs,), float("inf"), device="cuda")
                max_hold = torch.zeros(cfg.num_envs, dtype=torch.long, device="cuda")
                locked = None
                start_tcp = term.tcp_target.clone()
                oracle_stage = torch.zeros(cfg.num_envs, dtype=torch.long, device="cuda")
                last_target = term.tcp_target.clone()
                sat_sum = torch.zeros(cfg.num_envs, device="cuda")
                jump_sum = torch.zeros_like(sat_sum)
                late_count = torch.zeros_like(sat_sum)
                for step in range(math.ceil(cfg.episode_seconds / cfg.step_dt)):
                    raw = actor(obs).clone()
                    action = raw.clone()
                    requested = low + (raw[:, :3].clamp(-1, 1) + 1) * 0.5 * (high - low)
                    if mode.startswith("oracle_lift_first"):
                        _, _, _, goal = mdp.positions(env)
                        target = (
                            goal
                            - env.scene.env_origins
                            + torch.tensor(plug.GRIP_OFFSET, device="cuda")
                        )
                        physical, _ = term.plug_ik.manip.fk(env.sim.data.qpos[:, audit.qaddresses])
                        physical = term.plug_ik.manip_frame.pos_to_world(physical)
                        raised = (physical[:, 2] > 0.087) & (step * cfg.step_dt >= 1.24)
                        oracle_stage = torch.where((oracle_stage == 0) & raised, 1, oracle_stage)
                        aligned = ((physical[:, :2] - target[:, :2]).norm(dim=-1) < 0.003) & (
                            physical[:, 2] > 0.084
                        )
                        oracle_stage = torch.where((oracle_stage == 1) & aligned, 2, oracle_stage)
                        requested = target.clone()
                        requested[:, :2] = torch.where(
                            (oracle_stage == 0)[:, None], start_tcp[:, :2], target[:, :2]
                        )
                        requested[:, 2] = torch.where(
                            oracle_stage < 2,
                            0.09,
                            torch.maximum(target[:, 2], term.tcp_target[:, 2] - 0.0015),
                        )
                        action[:, :3] = 2 * (requested - low) / (high - low) - 1
                        if cfg.plug_wrist_yaw:
                            action[:, 3] = 0
                        if "noise" in mode:
                            scale = 0.1 if mode.endswith("noise10") else 1.0
                            noise = scale * torch.randn_like(action) * actor.distribution.std_param
                            if "delta5mm" in mode:
                                delta = ((requested - term.tcp_target) / 0.005).clamp(-1, 1)
                                delta = (delta + noise[:, :3]).clamp(-1, 1)
                                command = (term.tcp_target + 0.005 * delta).clamp(low, high)
                                action[:, :3] = 2 * (command - low) / (high - low) - 1
                            else:
                                action += noise
                    elif mode == "slew3mm":
                        requested = term.tcp_target + (requested - term.tcp_target).clamp(
                            -0.003, 0.003
                        )
                        action[:, :3] = 2 * (requested - low) / (high - low) - 1
                    elif mode == "hold_at_2s" and step * cfg.step_dt >= 2.0:
                        if locked is None:
                            pos, _ = term.plug_ik.manip.fk(env.sim.data.qpos[:, audit.qaddresses])
                            locked = term.plug_ik.manip_frame.pos_to_world(pos).clamp(low, high)
                        action[:, :3] = 2 * (locked - low) / (high - low) - 1
                        if cfg.plug_wrist_yaw:
                            action[:, 3] = 0
                    elif mode == "oracle_hold_at_2s" and step * cfg.step_dt >= 2.0:
                        _, _, _, goal = mdp.positions(env)
                        requested = (
                            goal
                            - env.scene.env_origins
                            + torch.tensor(plug.GRIP_OFFSET, device="cuda")
                        )
                        action[:, :3] = 2 * (requested - low) / (high - low) - 1
                        if cfg.plug_wrist_yaw:
                            action[:, 3] = 0
                    elif mode == "oracle_path":
                        _, _, _, goal = mdp.positions(env)
                        requested = (
                            goal
                            - env.scene.env_origins
                            + torch.tensor(plug.GRIP_OFFSET, device="cuda")
                        )
                        if step * cfg.step_dt < 1.55:
                            requested[:, 2] = 0.09
                        else:
                            requested[:, 2] = torch.maximum(
                                requested[:, 2], term.tcp_target[:, 2] - 0.0015
                            )
                        action[:, :3] = 2 * (requested - low) / (high - low) - 1
                        if cfg.plug_wrist_yaw:
                            action[:, 3] = 0
                    obs, reward, terminated, truncated, _ = env.step(action)
                    snap = audit.snapshot
                    live = ~finished
                    closest[live] = torch.minimum(
                        closest[live], snap["qpos_position_error_m"][live]
                    )
                    max_hold[live] = torch.maximum(max_hold[live], snap["success_hold_steps"][live])
                    late = live & (step * cfg.step_dt >= 1.5)
                    sat_sum += late * (raw[:, :3].abs() > 1).float().mean(-1)
                    jump_sum += late * (snap["target"] - last_target).norm(dim=-1)
                    late_count += late
                    if step % 1 == 0:
                        for i in [0, 1, 2, 3]:
                            if live[i]:
                                traces.append(
                                    dict(
                                        seed=seed,
                                        world=i,
                                        step=step,
                                        raw=raw[i].cpu().tolist(),
                                        action=action[i].cpu().tolist(),
                                        **{
                                            k: snap[k][i].cpu().tolist()
                                            for k in [
                                                "body",
                                                "goal",
                                                "target",
                                                "physical_tcp",
                                                "orientation",
                                                "object_velocity",
                                                "target_yaw",
                                            ]
                                        },
                                    )
                                )
                    last_target = snap["target"].clone()
                    ended = (terminated | truncated) & live
                    for i in ended.nonzero().flatten().tolist():
                        row = {k: v[i].cpu().tolist() for k, v in snap.items()}
                        row.update(
                            seed=seed,
                            world=i,
                            variant=cfg.variant_labels[i],
                            steps=step + 1,
                            closest_error=float(closest[i]),
                            max_hold=int(max_hold[i]),
                            late_action_saturation=float(sat_sum[i] / late_count[i].clamp_min(1)),
                            late_mean_target_step_m=float(jump_sum[i] / late_count[i].clamp_min(1)),
                        )
                        records.append(row)
                    finished |= ended
                    actor.reset(terminated | truncated)
                    if finished.all():
                        break
                assert finished.all()
        summary = {}
        for v in plug.VARIANTS:
            rows = [r for r in records if r["variant"] == v]
            summary[v] = dict(
                episodes=len(rows),
                successes=sum(r["success"] for r in rows),
                median_closest_mm=float(np.median([r["closest_error"] * 1000 for r in rows])),
                median_terminal_mm=float(
                    np.median([r["qpos_position_error_m"] * 1000 for r in rows])
                ),
                reached_2mm=sum(r["closest_error"] < 0.002 for r in rows),
                median_max_hold=float(np.median([r["max_hold"] for r in rows])),
                mean_action_saturation=float(np.mean([r["late_action_saturation"] for r in rows])),
                mean_target_step_mm=float(
                    np.mean([r["late_mean_target_step_m"] * 1000 for r in rows])
                ),
            )
        report = dict(
            checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            experiment=cfg.to_dict(),
            mode=mode,
            seeds=seeds,
            summary=summary,
            records=records,
            trace=traces,
            initial_hashes=initial_hashes,
            learner_updates=0,
            privileged=mode.startswith("oracle"),
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(mode=mode, checkpoint=str(checkpoint), summary=summary)), flush=True)
        return report
    finally:
        audit.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10000])
    parser.add_argument(
        "--modes", nargs="+", default=["baseline", "slew3mm", "hold_at_2s", "oracle_hold_at_2s"]
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    for name, checkpoint in manifest.items():
        checkpoint = Path(checkpoint)
        cfg = replace(
            saved_experiment(json.loads(checkpoint.with_name("experiment.json").read_text())),
            num_envs=128,
        )
        for mode in args.modes:
            path = args.output / f"{name}-{mode}.json"
            if path.exists():
                if not args.resume_completed:
                    raise FileExistsError(path)
                previous = json.loads(path.read_text())
                assert (
                    previous["checkpoint_sha256"]
                    == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                )
                assert previous["mode"] == mode and previous["seeds"] == args.seeds
                assert previous["experiment"] == json.loads(json.dumps(cfg.to_dict()))
                continue
            analyze(cfg, checkpoint, mode, args.seeds, path)


if __name__ == "__main__":
    main()
