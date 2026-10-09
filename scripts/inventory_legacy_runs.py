"""Recover actual historical plug run settings, excluding private W&B metadata."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=Path("../active_perception_arms"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/plug_analysis/legacy_run_inventory.json")
    )
    args = parser.parse_args()
    root = args.legacy_root.resolve()
    records = []
    for path in sorted((root / "wandb").glob("run-*/files/config.yaml")):
        config = yaml.safe_load(path.read_text())
        runner = config.get("train_cfg", {}).get("value", {})
        if not runner.get("run_name", "").startswith("pl3_"):
            continue
        env = config["env_cfg"]["value"]
        metadata = json.loads(path.with_name("wandb-metadata.json").read_text())
        summary_path = path.with_name("wandb-summary.json")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        run_dir = Path(config["log_dir"]["value"])
        final = run_dir / f"model_{runner['max_iterations'] - 1}.pt"
        checkpoint = {"path": str(final), "exists": final.exists()}
        if final.exists():
            weights = torch.load(final, map_location="cpu", weights_only=True)
            checkpoint.update(
                iteration_index=weights.get("iter"),
                sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
            )
        run_id = path.parent.parent.name.rsplit("-", 1)[1]
        commit = metadata.get("git", {}).get("commit")
        reward_path = "src/active_perception_arms/tasks/insertion/mdp/rewards.py"
        reward_source = subprocess.check_output(
            ["git", "show", f"{commit}:{reward_path}"], cwd=root
        )
        records.append(
            {
                "run_name": runner["run_name"],
                "wandb_id": run_id,
                "wandb_project": runner.get("wandb_project"),
                "saved_config": str(path),
                "saved_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_commit": commit,
                "reward_source_sha256": hashlib.sha256(reward_source).hexdigest(),
                "log_directory": str(run_dir),
                "final_checkpoint": checkpoint,
                "runner": {
                    key: runner.get(key)
                    for key in (
                        "seed",
                        "max_iterations",
                        "num_steps_per_env",
                        "multi_gpu",
                        "actor",
                        "critic",
                        "algorithm",
                        "obs_groups",
                    )
                },
                "environment": {
                    "num_envs_per_rank": env["scene"]["num_envs"],
                    "episode_seconds": env["episode_length_s"],
                    "physics_timestep": env["sim"]["mujoco"]["timestep"],
                    "decimation": env["decimation"],
                    "scale_rewards_by_dt": env["scale_rewards_by_dt"],
                    "finite_horizon": env["is_finite_horizon"],
                    "variants": sorted(env["scene"]["entities"]["object"].get("variants", {})),
                    "actions": env["actions"],
                    "rewards": env["rewards"],
                    "terminations": env["terminations"],
                    "reset": env["events"].get("reset_insertion"),
                    "sensors": env["scene"].get("sensors"),
                },
                "saved_training_summary": {
                    key: value
                    for key, value in summary.items()
                    if key
                    in (
                        "Train/mean_reward",
                        "Episode_Metrics/success_rate",
                        "Episode_Reward/insertion",
                        "Episode_Termination/success",
                    )
                },
                "scope": "Historical training summary, not independently reproduced held-out success. Checkpoint presence is not Slurm completion evidence. Different horizon, reward, architecture, simulation and possible task cues preclude a direct comparison.",
            }
        )
    result = {
        "legacy_root": str(root),
        "records": records,
        "verified_reward_at_saved_commit": "-log(100 * positional_error_m + 1), replaced by 1000 when instantaneous error < 2mm; no three-step hold",
        "training_budget_note": "1024 envs per rank, 24 steps, 2000 updates. Saved multi_gpu world size must be included in global transitions; do not equate per-rank49.152M with global total.",
        "independent_evaluation_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runs": len(records),
                "final_checkpoints": sum(r["final_checkpoint"]["exists"] for r in records),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
