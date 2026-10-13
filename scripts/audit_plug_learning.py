#!/usr/bin/env python3
"""Validate precision-control physics, reward stages and rendered PPO before training."""

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch
from mjlab.rl import RslRlVecEnvWrapper

from active_perception_arms import mdp, plug, plug_learning
from active_perception_arms.config import Experiment, static_candidates
from active_perception_arms.environment import make_env
from active_perception_arms.evaluate import load_actor
from active_perception_arms.evaluation_audit import TerminationAudit, summarize_termination_audit
from active_perception_arms.policy import CompactPPO, runner_cfg


def feasibility(cfg):
    env = make_env(cfg, "cuda:0")
    audit = TerminationAudit(env)
    records, traces = [], []
    try:
        with torch.inference_mode():
            for seed in range(11000, 11004):
                env.reset(seed=seed)
                audit.reset_batch()
                term = env.action_manager.get_term("arms")
                start = term.tcp_target.clone()
                stage = torch.zeros(cfg.num_envs, dtype=torch.long, device="cuda")
                finished = torch.zeros(cfg.num_envs, dtype=torch.bool, device="cuda")
                for step in range(math.ceil(cfg.episode_seconds / cfg.step_dt)):
                    tcp, _, _, goal = mdp.positions(env)
                    tcp = tcp - env.scene.env_origins
                    target = (
                        goal - env.scene.env_origins + torch.tensor(plug.GRIP_OFFSET, device="cuda")
                    )
                    stage = torch.where(
                        (stage == 0) & (tcp[:, 2] > 0.087) & (step * cfg.step_dt >= 1.24), 1, stage
                    )
                    aligned = ((tcp[:, :2] - target[:, :2]).norm(dim=-1) < 0.003) & (
                        tcp[:, 2] > 0.084
                    )
                    stage = torch.where((stage == 1) & aligned, 2, stage)
                    requested = target.clone()
                    requested[:, :2] = torch.where(
                        (stage == 0)[:, None], start[:, :2], target[:, :2]
                    )
                    requested[:, 2] = torch.where(
                        stage < 2, 0.09, torch.maximum(target[:, 2], term.tcp_target[:, 2] - 0.0015)
                    )
                    action = torch.zeros(cfg.num_envs, cfg.action_dim, device="cuda")
                    action[:, :3] = (
                        (requested - term.tcp_target) / plug_learning.STEP_METERS
                    ).clamp(-1, 1)
                    _, reward, terminated, truncated, _ = env.step(action)
                    assert torch.isfinite(reward).all()
                    if not finished[0]:
                        s = mdp.state(env)
                        traces.append(
                            dict(
                                seed=seed,
                                step=step,
                                reward=float(reward[0]),
                                raised=bool(s.plug_raised[0]),
                                aligned=bool(s.plug_aligned[0]),
                                error=float(audit.snapshot["qpos_position_error_m"][0]),
                            )
                        )
                    ended = (terminated | truncated) & ~finished
                    for i in ended.nonzero().flatten().tolist():
                        records.append(
                            dict(
                                seed=seed,
                                world=i,
                                variant=cfg.variant_labels[i],
                                steps=step + 1,
                                **{k: v[i].cpu().tolist() for k, v in audit.snapshot.items()},
                            )
                        )
                    finished |= ended
                    if finished.all():
                        break
                assert finished.all()
        result = summarize_termination_audit(records, "plug")
        for key, value in result.items():
            if key.startswith("success_") and "violation" in key:
                assert value in (None, 0), (key, value)
        result.update(
            experiment=cfg.to_dict(),
            controller="Privileged lift-first oracle; not a learned policy",
            records=records,
            representative_reward_traces=traces,
            per_variant={
                v: dict(
                    episodes=sum(r["variant"] == v for r in records),
                    successes=sum(r["success"] and r["variant"] == v for r in records),
                )
                for v in plug.VARIANTS
            },
        )
        result["passed"] = all(r["successes"] >= 122 for r in result["per_variant"].values())
        return result
    finally:
        audit.close()
        env.close()


def smoke(cfg, output):
    env = make_env(cfg, "cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env)
        obs, _ = wrapped.reset()
        options = asdict(runner_cfg(cfg, iterations=2))
        options["actor"]["distribution_cfg"]["init_std"] = 0.25
        options["algorithm"]["entropy_coef"] = 0.001
        options["multi_gpu"] = None
        for name in ("actor", "critic"):
            options[name] = {k: v for k, v in options[name].items() if v is not None}
        alg = CompactPPO.construct_algorithm(obs, wrapped, options, "cuda:0")
        before = {k: v.clone() for k, v in alg.actor.cnns.state_dict().items()}
        metrics = []
        for _ in range(2):
            with torch.inference_mode():
                for _ in range(24):
                    action = alg.act(obs)
                    obs, rewards, dones, info = wrapped.step(action)
                    assert torch.isfinite(rewards).all() and torch.isfinite(action).all()
                    for name in cfg.sensors:
                        assert obs[name].dtype == torch.uint8 and obs[name].shape == (8, 3, 96, 128)
                        assert (
                            obs[name].flatten(1).max(1).values > obs[name].flatten(1).min(1).values
                        ).all()
                    alg.process_env_step(obs, rewards, dones, info)
                alg.compute_returns(obs)
            metric = alg.update()
            assert all(math.isfinite(v) for v in metric.values())
            metrics.append(metric)
        for name in cfg.sensors:
            assert any(
                not torch.equal(v, before[k])
                for k, v in alg.actor.cnns.state_dict().items()
                if k.startswith(name)
            )
        output.mkdir(parents=True)
        (output / "experiment.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        (output / "runner.json").write_text(json.dumps(options, indent=2))
        torch.save({"actor_state_dict": alg.actor.state_dict()}, output / "model.pt")
        restored = load_actor(output / "model.pt", obs, "cuda:0")
        with torch.inference_mode():
            alg.actor.reset()
            restored.reset()
            torch.testing.assert_close(restored(obs), alg.actor(obs))
        return dict(
            passed=True,
            experiment=cfg.to_dict(),
            updates=2,
            transitions=384,
            metrics=metrics,
            all_rgb_encoders_updated=True,
            checkpoint_roundtrip=True,
        )
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    base = Experiment(
        condition="wrist_static",
        num_envs=128,
        fixed_position=static_candidates("plug")[7],
        plug_control="delta5mm",
        reward_profile="plug_staged",
        render_sensors=False,
    )
    result = dict(gpu=torch.cuda.get_device_name(), feasibility=feasibility(base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result["smoke"] = {}
    for condition in ("wrist", "wrist_static", "active"):
        cfg = replace(base, num_envs=8, condition=condition, render_sensors=True)
        result["smoke"][condition] = smoke(cfg, args.output.parent / f"smoke-{condition}")
    result["passed"] = result["feasibility"]["passed"] and all(
        s["passed"] for s in result["smoke"].values()
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "per_variant": result["feasibility"]["per_variant"],
                "smoke": list(result["smoke"]),
            }
        ),
        flush=True,
    )
    assert result["passed"]


if __name__ == "__main__":
    main()
