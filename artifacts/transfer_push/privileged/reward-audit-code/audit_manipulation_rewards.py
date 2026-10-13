"""Compare real-physics returns of successful and stalled diagnostic controllers."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from audit_manipulation_study import PushController
from audit_transfer_warp import BatchedTransferController

from active_perception_arms.config import Experiment
from active_perception_arms.environment import make_env
from active_perception_arms.evaluation_audit import TerminationAudit, summarize_termination_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    report = dict(
        meaning="Scripted reward sanity check, not learned competence", gamma=0.99, tasks={}
    )
    for task in ("transfer", "push"):
        cfg = Experiment(
            task=task,
            condition="wrist_static",
            actor_observation="privileged",
            manip_control="cartesian",
            task_revision="pick_place_v2" if task == "transfer" else "push_v2",
            reward_profile=task + "_dense",
            reset_xy_jitter=0.02,
            num_envs=64,
            seed=42000,
            occlusion="dynamic",
            memory="none",
        )
        env = make_env(cfg, "cuda:0")
        audit = TerminationAudit(env)
        cases = (
            ("complete", "reach_only", "grasp_hold", "lift_hold")
            if task == "transfer"
            else ("complete", "behind_only", "goal_hold")
        )
        results = []
        try:
            for case in cases:
                torch.manual_seed(cfg.seed)
                env.reset()
                audit.reset_batch()
                initial = hashlib.sha256(
                    env.sim.data.qpos.cpu().numpy().tobytes()
                    + env.sim.data.mocap_pos.cpu().numpy().tobytes()
                ).hexdigest()
                control = (
                    BatchedTransferController(env) if task == "transfer" else PushController(env)
                )
                done = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
                total = torch.zeros(env.num_envs, device=env.device)
                discounted = total.clone()
                records = []
                for step in range(round(cfg.episode_seconds / cfg.step_dt)):
                    if task == "transfer" and case != "complete":
                        control.stage.clamp_(
                            max={"reach_only": 1, "grasp_hold": 2, "lift_hold": 3}[case]
                        )
                    action = control()
                    term = env.action_manager.get_term("arms")
                    if task == "push" and case != "complete":
                        from active_perception_arms import mdp

                        tcp, obj, _, goal = mdp.positions(env)
                        direction = goal[:, :2] - obj[:, :2]
                        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        target = obj.clone()
                        target[:, :2] -= direction * (0.025 if case == "behind_only" else 0.017)
                        target[:, 2] = env.scene.env_origins[:, 2] + 0.040
                        action[:, :3] = (
                            2
                            * (target - env.scene.env_origins - term.cartesian.low)
                            / (term.cartesian.high - term.cartesian.low)
                            - 1
                        )
                    _, reward, terminated, truncated, _ = env.step(action)
                    assert torch.isfinite(reward).all()
                    total += torch.where(done, 0, reward)
                    discounted += torch.where(done, 0, reward) * (0.99**step)
                    ended = (terminated | truncated) & ~done
                    for index in ended.nonzero().flatten().tolist():
                        row = {k: v[index].cpu().tolist() for k, v in audit.snapshot.items()}
                        row.update(
                            index=index,
                            steps=step + 1,
                            episode_return=total[index].item(),
                            discounted_return=discounted[index].item(),
                        )
                        records.append(row)
                    done |= ended
                    if done.all():
                        break
                assert done.all()
                result = summarize_termination_audit(records, task)
                result.update(
                    case=case,
                    initial_physics_sha256=initial,
                    mean_return=total.mean().item(),
                    mean_discounted_return=discounted.mean().item(),
                )
                results.append(result)
                report["tasks"][task] = dict(experiment=cfg.to_dict(), cases=results)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        dict(
                            task=task,
                            case=case,
                            successes=result["terminal_flags"]["success"],
                            mean_return=result["mean_return"],
                            mean_discounted_return=result["mean_discounted_return"],
                        )
                    ),
                    flush=True,
                )
            assert len({r["initial_physics_sha256"] for r in results}) == 1
        finally:
            audit.close()
            env.close()


if __name__ == "__main__":
    main()
