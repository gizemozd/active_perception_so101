"""Replay validation seeds at the same batch size; record first success/failure per variant."""

import argparse
import json
import math
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from . import mdp
from .config import saved_experiment
from .environment import make_env
from .evaluate import load_actor
from .native import NativeEnv
from .sanity import OverviewRecorder, mosaic
from .warp_sanity import sync_native_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    cfg = saved_experiment(report["experiment"])
    checkpoint = Path(report["checkpoint"])
    selected = {}
    for i, (variant, won) in enumerate(zip(report["variants"], report["outcomes"], strict=True)):
        selected.setdefault((variant, bool(won)), i)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    env = make_env(cfg, args.device)
    records = []
    try:
        obs, _ = env.reset(seed=report["seed"])
        actor = load_actor(checkpoint, obs, args.device)
        for batch in sorted({i // cfg.num_envs for i in selected.values()}):
            indices = {
                i % cfg.num_envs: (v, won, i)
                for (v, won), i in selected.items()
                if i // cfg.num_envs == batch
            }
            obs, _ = env.reset(seed=report["seed"] + batch)
            actor.reset()
            finished = torch.zeros(cfg.num_envs, device=args.device, dtype=torch.bool)
            actual_won = torch.zeros_like(finished)
            with ExitStack() as stack:
                media = {}
                for index, (variant, won, episode) in indices.items():
                    name = f"{variant}_{'success' if won else 'failure'}_episode{episode}"
                    proxy = NativeEnv(replace(cfg, num_envs=1, plug_variant=variant))
                    overview_path = args.output / f"{name}_outside.mp4"
                    policy_path = args.output / f"{name}_policy.mp4"
                    observer = OverviewRecorder(proxy.model, overview_path, fps=1 / cfg.step_dt)
                    stack.callback(observer.close)
                    writer = stack.enter_context(
                        imageio.get_writer(policy_path, fps=1 / cfg.step_dt, macro_block_size=2)
                    )
                    record = {
                        "episode": episode,
                        "variant": variant,
                        "expected_success": won,
                        "outside_video": str(overview_path),
                        "policy_video": str(policy_path),
                        "steps": [],
                    }
                    records.append(record)
                    media[index] = (proxy, observer, writer, record)
                with torch.inference_mode():
                    for step in range(math.ceil(cfg.episode_seconds / cfg.step_dt)):
                        action = actor(obs)
                        _, obj, _, goal = mdp.positions(env)
                        errors = torch.linalg.vector_norm(obj - goal, dim=-1)
                        for index, (proxy, observer, writer, record) in media.items():
                            if finished[index]:
                                continue
                            sync_native_state(proxy, env, step, index)
                            observer.capture(proxy.data)
                            frames = [
                                obs[name][index].permute(1, 2, 0).cpu().numpy()
                                for name in cfg.sensors
                            ]
                            tile = mosaic(
                                frames,
                                [
                                    f"{name}: actual input t={step * cfg.step_dt:.2f}s"
                                    for name in cfg.sensors
                                ],
                            )
                            writer.append_data(np.asarray(tile))
                            record["steps"].append(
                                {
                                    "time": step * cfg.step_dt,
                                    "position_error_m": errors[index].item(),
                                    "action": action[index].cpu().tolist(),
                                }
                            )
                        obs, reward, term, trunc, _ = env.step(action)
                        done = term | trunc
                        actual_won |= done & ~finished & env.termination_manager.get_term("success")
                        for index, (_, _, _, record) in media.items():
                            if not finished[index]:
                                record["steps"][-1]["reward"] = reward[index].item()
                        finished |= done
                        actor.reset(done)
                    for index, (_, _, _, record) in media.items():
                        record["reproduced_success"] = bool(actual_won[index].item())
                        assert record["reproduced_success"] == record["expected_success"], (
                            "Validation replay differs"
                        )
    finally:
        env.close()
    out = args.output / "representatives.json"
    out.write_text(
        json.dumps(
            {
                "selection": "First success and first failure per variant, when present",
                "evaluation_report": str(args.report),
                "records": records,
            },
            indent=2,
        )
    )
    if args.wandb:
        import wandb

        parent = json.loads(checkpoint.with_name("wandb_run.json").read_text())
        run = wandb.init(
            project=parent["project"], entity=parent["entity"], id=report["wandb_id"], resume="must"
        )
        for record in records:
            label = f"{record['variant']}_{'success' if record['expected_success'] else 'failure'}"
            run.log(
                {
                    f"validation/videos/{label}_{kind}": wandb.Video(record[key], format="mp4")
                    for kind, key in (("outside", "outside_video"), ("policy", "policy_video"))
                }
            )
        run.save(str(out), base_path=str(args.output))
        run.finish()


if __name__ == "__main__":
    main()
