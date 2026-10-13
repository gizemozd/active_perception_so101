"""Validate Cartesian transfer/push physics, RGB learning and checkpoint reload on GPU."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from audit_transfer_warp import BatchedTransferController, ppo_smoke

from active_perception_arms import mdp
from active_perception_arms.config import Experiment
from active_perception_arms.environment import make_env
from active_perception_arms.evaluation_audit import TerminationAudit, summarize_termination_audit


class PushController:
    def __init__(self, env):
        self.env = env
        self.retreating = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.retreat = torch.zeros(env.num_envs, 3, device=env.device)

    def __call__(self):
        env = self.env
        tcp, obj, _, goal = mdp.positions(env)
        direction = goal[:, :2] - obj[:, :2]
        distance = direction.norm(dim=-1)
        direction /= distance[:, None].clamp_min(1e-6)
        p = obj.clone()
        p[:, :2] -= direction * 0.017
        p[:, 2] = env.scene.env_origins[:, 2] + 0.040
        newly = (distance < 0.009) & ~self.retreating
        self.retreat[newly] = tcp[newly] + tcp.new_tensor([0, 0, 0.065])
        self.retreating |= newly
        p = torch.where(self.retreating[:, None], self.retreat, p)
        term = env.action_manager.get_term("arms")
        action = torch.zeros(env.num_envs, term.exp.action_dim, device=env.device)
        action[:, :3] = (
            2
            * (p - env.scene.env_origins - term.cartesian.low)
            / (term.cartesian.high - term.cartesian.low)
            - 1
        )
        return action


def audit(cfg, output):
    env = make_env(cfg, "cuda:0")
    audit = TerminationAudit(env)
    try:
        env.reset()
        audit.reset_batch()
        controller = (
            BatchedTransferController(env) if cfg.task == "transfer" else PushController(env)
        )
        finished = torch.zeros(cfg.num_envs, dtype=torch.bool, device=env.device)
        records = []
        for step in range(round(cfg.episode_seconds / cfg.step_dt)):
            _, reward, terminated, truncated, _ = env.step(controller())
            assert torch.isfinite(reward).all()
            ended = (terminated | truncated) & ~finished
            for i in ended.nonzero().flatten().tolist():
                row = {k: v[i].cpu().tolist() for k, v in audit.snapshot.items()}
                row.update(index=i, steps=step + 1)
                records.append(row)
            finished |= ended
            if finished.all():
                break
        assert finished.all()
        result = summarize_termination_audit(records, cfg.task)
        result["experiment"] = cfg.to_dict()
        assert result["success_hold_violations"] == 0
        assert result["success_" + cfg.task + "_violations"] == 0
        output.write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        audit.close()
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--jitter", type=float, default=0.02)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    report = {}
    for task in ("transfer", "push"):
        cfg = Experiment(
            task=task,
            condition="wrist_static",
            occlusion="clean",
            num_envs=args.num_envs,
            seed=10000,
            task_revision="pick_place_v2" if task == "transfer" else "push_v2",
            reward_profile=task + "_dense",
            manip_control="cartesian",
            reset_xy_jitter=args.jitter,
        )
        result = audit(cfg, args.output / (task + "-physics.json"))
        smoke = ppo_smoke("cuda:0", args.output / (task + "-smoke"), replace(cfg, num_envs=8))
        report[task] = dict(
            successes=result["terminal_flags"]["success"],
            episodes=len(result["records"]),
            smoke=smoke,
        )
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report[task]), flush=True)


if __name__ == "__main__":
    main()
