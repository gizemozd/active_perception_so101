#!/usr/bin/env python3
"""Bounded transfer feasibility audit with real terminations and rendered PPO smoke."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.rl import RslRlVecEnvWrapper

from active_perception_arms import mdp
from active_perception_arms.capture_evaluation import RepresentativeCapture, render_capture
from active_perception_arms.config import Experiment
from active_perception_arms.environment import make_env
from active_perception_arms.evaluate import load_actor
from active_perception_arms.evaluation_audit import TerminationAudit, summarize_termination_audit
from active_perception_arms.policy import CompactPPO, runner_cfg
from active_perception_arms.robots.kinematics import grasp_site_rotation


class BatchedTransferController:
    """Same privileged stages as Native ScriptedPolicy, batched only for this audit."""

    def __init__(self, env):
        self.env, self.s = env, mdp.state(env)
        self.cfg = self.s.cfg
        self.stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.ticks = torch.zeros_like(self.stage)
        self.start = mdp.positions(env)[1].clone()
        self.qseed = env.action_manager.get_term("arms").targets["manipulator"][:, :5].clone()
        self.rotation = torch.as_tensor(
            grasp_site_rotation(env.sim.mj_model, prefix="manipulator/", yaw=-np.pi / 2),
            device=env.device,
            dtype=torch.float32,
        )[None].expand(env.num_envs, -1, -1)

        # Compile one IK iteration; a 12-iteration unrolled graph makes a short audit
        # unnecessarily compilation-bound. The controller is never used for training.
        def iteration(p, q):
            return self.s.tcp_chain.solve_ik(p, q, target_rot=self.rotation, iters=1, damping=0.006)

        self.ik_iteration = (
            torch.compile(iteration, fullgraph=True)
            if str(env.device).startswith("cuda")
            else iteration
        )

    def __call__(self):
        env, cfg, stage = self.env, self.cfg, self.stage
        tcp, obj, _, goal = mdp.positions(env)
        p = self.start.clone()
        p[:, 2] += torch.where(stage == 0, 0.055, 0.003)
        lift = obj.clone()
        lift[:, 2] = env.scene.env_origins[:, 2] + 0.110
        p = torch.where((stage == 3)[:, None], lift, p)
        approach = goal.clone()
        approach[:, 0] -= 0.035
        approach[:, 2] = env.scene.env_origins[:, 2] + 0.110
        p = torch.where((stage == 4)[:, None], approach, p)
        lower = goal.clone()
        lower[:, 2] += 0.004
        p = torch.where((stage == 5)[:, None], lower, p)
        p = torch.where((stage == 6)[:, None], goal, p)
        retreat = goal + torch.tensor([-0.065, 0, 0.055], device=env.device)
        p = torch.where((stage >= 7)[:, None], retreat, p)
        p[:, 1] -= 0.008
        action = torch.zeros(env.num_envs, cfg.action_dim, device=env.device)
        term = env.action_manager.get_term("arms")
        if cfg.manip_control == "cartesian":
            low, high = term.cartesian.low, term.cartesian.high
            action[:, :3] = 2 * (p - env.scene.env_origins - low) / (high - low) - 1
        else:
            local = p - env.scene.env_origins - self.s.manip_base
            for _ in range(12):
                self.qseed = self.ik_iteration(local, self.qseed)
            target = term.targets["manipulator"][:, :5]
            action[:, :5] = ((self.qseed - target) / cfg.joint_step).clamp(-1, 1)
        action[:, cfg.manip_dim - 1] = torch.where(
            stage < 2, 0.5, torch.where(stage >= 6, 0.65, -1.0)
        )
        moving = env.episode_length_buf * cfg.step_dt >= cfg.initial_seconds
        self.ticks += moving
        close = (tcp - p).norm(dim=-1) < 0.006
        force_dwell = (stage == 2) | (stage == 6)
        advance = moving & (close | force_dwell) & (self.ticks > torch.where(force_dwell, 22, 10))
        self.stage = torch.where(advance, (stage + 1).clamp(max=7), stage)
        self.ticks = torch.where(advance, 0, self.ticks)
        return action


def feasibility(cfg, device, directory):
    env = make_env(cfg, device)
    audit = TerminationAudit(env)
    capture = RepresentativeCapture(directory / "capture", cfg) if cfg.render_sensors else None
    try:
        obs, _ = env.reset()
        audit.reset_batch()
        controller = BatchedTransferController(env)
        finished = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
        won = finished.clone()
        elapsed = torch.zeros(cfg.num_envs, device=device)
        records = []
        reward_sum = torch.zeros(cfg.num_envs, device=device)
        for step in range(round(cfg.episode_seconds / cfg.step_dt)):
            action = controller()
            if capture:
                capture.step(env, obs, action)
            obs, reward, terminated, truncated, _ = env.step(action)
            assert torch.isfinite(reward).all()
            reward_sum += torch.where(finished, 0.0, reward)
            if capture:
                capture.reward(reward)
                for name in cfg.sensors:
                    assert obs[name].dtype == torch.uint8
                    assert obs[name].shape == (cfg.num_envs, 3, cfg.height, cfg.width)
            ended = (terminated | truncated) & ~finished
            for index in ended.nonzero().flatten().tolist():
                row = {k: v[index].cpu().tolist() for k, v in audit.snapshot.items()}
                row.update(
                    index=index,
                    steps=step + 1,
                    stage=int(controller.stage[index]),
                    episode_return=float(reward_sum[index]),
                )
                records.append(row)
                won[index] = row["success"]
                elapsed[index] = (step + 1) * cfg.step_dt
            finished |= ended
            if finished.all():
                break
        assert finished.all(), "Every sampled first episode must be accounted for"
        result = summarize_termination_audit(records, "transfer")
        result.update(
            experiment=cfg.to_dict(),
            backend="MuJoCo Warp",
            device=device,
            controller="privileged diagnostic; no learned policy",
            training_updates=0,
        )
        if capture:
            result["representative_capture"] = capture.finish_batch(0, won, elapsed, cfg.num_envs)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        assert result["success_hold_violations"] == 0
        assert result["success_transfer_violations"] == 0
        return result
    finally:
        audit.close()
        env.close()


def ppo_smoke(device, directory, cfg=None):
    cfg = cfg or Experiment(
        task="transfer", condition="wrist_static", occlusion="clean", num_envs=8
    )
    env = make_env(cfg, device)
    try:
        wrapped = RslRlVecEnvWrapper(env)
        obs, _ = wrapped.reset()
        options = asdict(runner_cfg(cfg, iterations=2))
        options["multi_gpu"] = None
        for name in ("actor", "critic"):
            options[name] = {k: v for k, v in options[name].items() if v is not None}
        alg = CompactPPO.construct_algorithm(obs, wrapped, options, device)
        before = {k: v.clone() for k, v in alg.actor.cnns.state_dict().items()}
        metrics = []
        for _ in range(2):
            with torch.inference_mode():
                for _ in range(24):
                    action = alg.act(obs)
                    obs, reward, dones, info = wrapped.step(action)
                    assert torch.isfinite(reward).all() and torch.isfinite(action).all()
                    assert obs["critic"].dtype == torch.float32
                    for camera in cfg.sensors:
                        rgb = obs[camera]
                        assert rgb.shape == (8, 3, 72, 96) and rgb.dtype == torch.uint8
                        assert (rgb.flatten(1).amax(1) > rgb.flatten(1).amin(1)).all()
                    alg.process_env_step(obs, reward, dones, info)
                alg.compute_returns(obs)
            metric = alg.update()
            assert all(math.isfinite(v) for v in metric.values())
            metrics.append(metric)
        for camera in cfg.sensors:
            assert any(
                not torch.equal(v, before[k])
                for k, v in alg.actor.cnns.state_dict().items()
                if k.startswith(camera)
            )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "experiment.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        (directory / "runner.json").write_text(json.dumps(options, indent=2))
        checkpoint = directory / "model.pt"
        torch.save({"actor_state_dict": alg.actor.state_dict()}, checkpoint)
        restored = load_actor(checkpoint, obs, device)
        # Compare from identical empty GRU state, not the training rollout's state.
        with torch.inference_mode():
            alg.actor.reset(torch.ones(8, dtype=torch.bool, device=device))
            torch.testing.assert_close(restored(obs), alg.actor(obs))
        return dict(
            passed=True,
            updates=2,
            transitions=384,
            both_rgb_encoders_updated=True,
            checkpoint_roundtrip=True,
            metrics=metrics,
            competence_claim=False,
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[10000, 10001])
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--render-videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4 if args.device.startswith("cuda") else 1)
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "Allocated CUDA GPU required"
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        cfg = Experiment(
            task="transfer",
            condition="wrist_static",
            occlusion="clean",
            seed=seed,
            num_envs=args.num_envs,
            render_sensors=not args.no_render,
        )
        result = feasibility(cfg, args.device, args.output / f"seed-{seed}")
        results.append(result)
        print(json.dumps({"seed": seed, "terminals": result["terminal_flags"]}), flush=True)
    smoke = None if args.skip_smoke else ppo_smoke(args.device, args.output / "ppo-smoke")
    report = dict(
        cases=results,
        ppo_smoke=smoke,
        competence_claim=False,
        gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.render_videos:
        for i, result in enumerate(results):
            render_capture(result, args.output / f"videos-{i}")


if __name__ == "__main__":
    main()
