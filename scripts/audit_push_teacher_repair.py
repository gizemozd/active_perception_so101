"""Physical GPU audit of a staged push expert; no learned-policy success claim."""

import argparse
import json
from pathlib import Path

import torch

from active_perception_arms import mdp
from active_perception_arms.config import Experiment
from active_perception_arms.environment import make_env
from active_perception_arms.evaluation_audit import TerminationAudit, summarize_termination_audit


class PushTeacher:
    def __init__(self, env, contact_offset=0.017):
        self.env = env
        self.contact_offset = contact_offset
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.initial = env.action_manager.get_term("arms").tcp_target.clone()
        self.retreat = self.initial.clone()

    def __call__(self):
        env = self.env
        term = env.action_manager.get_term("arms")
        fresh = env.episode_length_buf <= 1
        self.phase[fresh] = 0
        self.initial[fresh] = term.tcp_target[fresh]
        tcp, obj, _, goal = mdp.positions(env)
        origins = env.scene.env_origins
        tcp = tcp - origins
        obj = obj - origins
        goal = goal - origins
        d = goal[:, :2] - obj[:, :2]
        distance = d.norm(dim=-1)
        direction = d / distance[:, None].clamp_min(1e-6)
        behind = obj[:, :2] - 0.035 * direction
        time = env.episode_length_buf * env.step_dt
        self.phase = torch.where(
            (self.phase == 0) & (tcp[:, 2] > 0.068) & (time > 1.2), 1, self.phase
        )
        self.phase = torch.where(
            (self.phase == 1) & ((tcp[:, :2] - behind).norm(dim=-1) < 0.003), 2, self.phase
        )
        self.phase = torch.where((self.phase == 2) & (tcp[:, 2] < 0.043), 3, self.phase)
        new = (self.phase == 3) & (distance < 0.003)
        self.retreat[new] = tcp[new]
        self.retreat[new, 2] = 0.075
        self.phase = torch.where(new, 4, self.phase)
        target = tcp.clone()
        target[:, :2] = torch.where((self.phase == 0)[:, None], self.initial[:, :2], behind)
        target[:, 2] = torch.where(self.phase < 2, 0.075, 0.037)
        target[:, :2] = torch.where(
            (self.phase == 3)[:, None], obj[:, :2] - self.contact_offset * direction, target[:, :2]
        )
        target = torch.where((self.phase == 4)[:, None], self.retreat, target)
        action = torch.zeros(env.num_envs, term.exp.action_dim, device=env.device)
        action[:, :3] = (
            2 * (target - term.cartesian.low) / (term.cartesian.high - term.cartesian.low) - 1
        )
        return action.clamp(-1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizon", type=float, default=20)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--contact-offset", type=float, default=0.017)
    p.add_argument("--wide", action="store_true")
    p.add_argument("--clear", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    cfg = Experiment(
        task="push",
        task_revision="push_v2",
        push_target_layout="arc_v2",
        condition="wrist_static",
        occlusion="clean",
        reward_profile="push_dense",
        manip_control="cartesian",
        reset_xy_jitter=0.02,
        num_envs=args.num_envs,
        seed=31000,
        episode_seconds=args.horizon,
        render_sensors=False,
        push_control_revision="wide_v2" if args.wide else "legacy_v1",
        push_workspace_revision="arc_clear_v2" if args.clear else "partition_v1",
    )
    env = make_env(cfg, "cuda:0")
    audit = TerminationAudit(env)
    try:
        env.reset()
        audit.reset_batch()
        teacher = PushTeacher(env, args.contact_offset)
        _, obj, _, goal = mdp.positions(env)
        initial_obj = (obj - env.scene.env_origins).clone()
        initial_goal = (goal - env.scene.env_origins).clone()
        finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        records = []
        with torch.inference_mode():
            for step in range(round(cfg.episode_seconds / cfg.step_dt)):
                _, reward, terminated, truncated, _ = env.step(teacher())
                ended = (terminated | truncated) & ~finished
                for i in ended.nonzero().flatten().tolist():
                    row = {k: v[i].cpu().tolist() for k, v in audit.snapshot.items()}
                    row.update(
                        index=i,
                        steps=step + 1,
                        phase=int(teacher.phase[i]),
                        initial_object=initial_obj[i].tolist(),
                        initial_goal=initial_goal[i].tolist(),
                    )
                    records.append(row)
                finished |= ended
                if finished.all():
                    break
        result = summarize_termination_audit(records, cfg.task)
        result["experiment"] = cfg.to_dict()
        result["contact_offset"] = args.contact_offset
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result["terminal_flags"], flush=True)
    finally:
        audit.close()
        env.close()


if __name__ == "__main__":
    main()
