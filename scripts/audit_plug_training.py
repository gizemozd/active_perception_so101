"""Audit retained plug runs without training, resimulation, or changing checkpoints."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def metric_window(rows, key, start, end):
    values = [
        (r["iteration"], float(r[key])) for r in rows if start <= r["iteration"] <= end and key in r
    ]
    if not values:
        return None
    x, y = np.asarray(values).T
    slope = float(np.polyfit(x - x[0], y, 1)[0]) if len(x) > 1 else 0.0
    return {
        "start_update": int(x[0]),
        "end_update": int(x[-1]),
        "samples": len(values),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "slope_per_100_updates": slope * 100,
        "fitted_change_over_window": float(slope * (x[-1] - x[0])),
    }


def audit_history(rows, updates, transitions_per_update):
    expected = list(range(1, updates + 1))
    iterations = [r["iteration"] for r in rows]
    transitions_ok = all(
        r["transitions"] == r["iteration"] * transitions_per_update
        and r["iteration_transitions"] == transitions_per_update
        for r in rows
    )
    metrics = [
        "train/Train/mean_reward",
        "train/Episode_Reward/task",
        "train/Train/mean_episode_length",
        "train/Episode_Metrics/success_rate",
        "train/Loss/value",
        "train/Loss/surrogate",
        "train/Loss/entropy",
        "train/Loss/learning_rate",
        "train/Policy/mean_std",
    ]
    windows = {}
    for key in metrics:
        windows[key] = {
            label: metric_window(rows, key, start, end)
            for label, start, end in (
                ("early_100", 1, 100),
                ("mid_100", 701, 800),
                ("late_100", updates - 99, updates),
                ("late_300", updates - 299, updates),
                ("penultimate_150", updates - 299, updates - 150),
                ("final_150", updates - 149, updates),
            )
        }
    reward = windows["train/Train/mean_reward"]
    early, late = reward["early_100"], reward["late_100"]
    tail = reward["late_300"]
    final, preceding = reward["final_150"], reward["penultimate_150"]
    denominator = max(abs(tail["mean"]), 0.1)
    mean_change = (final["mean"] - preceding["mean"]) / denominator
    fitted_change = tail["fitted_change_over_window"] / denominator
    plateau = abs(mean_change) <= 0.10 and abs(fitted_change) <= 0.10
    late_lr = [r["train/Loss/learning_rate"] for r in rows[-300:]]
    return {
        "history_records": len(rows),
        "expected_records": updates,
        "contiguous_unique_updates": iterations == expected,
        "transition_accounting_correct": transitions_ok,
        "last_update": iterations[-1] if rows else None,
        "last_transitions": rows[-1]["transitions"] if rows else None,
        "nonfinite_logged_metrics": sum(
            not np.isfinite(v)
            for r in rows
            for k, v in r.items()
            if k.startswith("train/") and isinstance(v, (int, float))
        ),
        "metric_windows": windows,
        "reward_early_to_late_gain": late["mean"] - early["mean"],
        "reward_plateau": {
            "descriptive_plateau": plateau,
            "final_150_vs_previous_150_fraction": mean_change,
            "last_300_fitted_change_fraction": fitted_change,
            "rule": "Both absolute fractional changes <= 0.10, denominator max(abs(last-300 mean),0.1). Descriptive threshold, not a statistical stationarity test.",
        },
        "lr_floor_updates_last_300": sum(v <= 1.0000001e-5 for v in late_lr),
        "lr_max_all_updates": max(r["train/Loss/learning_rate"] for r in rows),
    }


def compact_evaluation(path):
    d = json.loads(path.read_text())
    return {
        "path": str(path),
        **{
            k: d[k]
            for k in (
                "checkpoint",
                "split",
                "seed",
                "training_seed",
                "episodes",
                "successes",
                "success_rate",
                "wilson_95",
                "per_variant",
                "mean_episode_seconds",
                "mean_success_completion_seconds",
                "mean_camera_joint_travel_rad",
                "wandb_url",
            )
            if k in d
        },
    }


def historical_comparison(root):
    """Read real retained run configs safely; YAML Python tags are never executed."""
    if not root.exists():
        return {"retained_configs_found": False}
    import yaml

    runs = [
        root / "logs/rsl_rl/insertion_vision/2026-08-03_01-58-14_pl3_static_s0",
        root / "logs/rsl_rl/insertion_active_vision/2026-08-03_04-48-50_pl3_active_s0",
    ]
    saved = []
    for directory in runs:
        agent_path, env_path = directory / "params/agent.yaml", directory / "params/env.yaml"
        if not agent_path.exists() or not env_path.exists():
            continue
        agent = yaml.load(agent_path.read_text(), Loader=yaml.BaseLoader)
        env = yaml.load(env_path.read_text(), Loader=yaml.BaseLoader)
        env_node = yaml.compose(env_path.read_text(), Loader=yaml.BaseLoader)

        def callable_tag(*keys, env_node=env_node):
            node = env_node
            for key in keys:
                node = next(value for name, value in node.value if name.value == key)
            return node.tag.removeprefix("tag:yaml.org,2002:python/name:")

        saved.append(
            {
                "directory": str(directory),
                "agent_yaml_sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest(),
                "env_yaml_sha256": hashlib.sha256(env_path.read_bytes()).hexdigest(),
                "seed": int(agent["seed"]),
                "num_envs": int(env["scene"]["num_envs"]),
                "max_iterations": int(agent["max_iterations"]),
                "configured_transitions": int(agent["max_iterations"])
                * int(agent["num_steps_per_env"])
                * int(env["scene"]["num_envs"]),
                "episode_seconds": float(env["episode_length_s"]),
                "timestep": float(env["sim"]["mujoco"]["timestep"]),
                "decimation": int(env["decimation"]),
                "step_dt": float(env["sim"]["mujoco"]["timestep"]) * int(env["decimation"]),
                "sensor_resolution": {
                    s["name"]: [int(s["width"]), int(s["height"])] for s in env["scene"]["sensors"]
                },
                "actor_hidden_dims": [int(v) for v in agent["actor"]["hidden_dims"]],
                "actor_rnn_type": agent["actor"]["rnn_type"],
                "critic_hidden_dims": [int(v) for v in agent["critic"]["hidden_dims"]],
                "critic_rnn_type": agent["critic"]["rnn_type"],
                "init_std": float(agent["actor"]["distribution_cfg"]["init_std"]),
                "algorithm": {
                    key: agent["algorithm"][key]
                    for key in (
                        "learning_rate",
                        "schedule",
                        "entropy_coef",
                        "num_learning_epochs",
                        "num_mini_batches",
                        "gamma",
                        "lam",
                    )
                },
                "saved_reward_callable": callable_tag("rewards", "insertion", "func"),
                "saved_success_callable": callable_tag("terminations", "success", "func"),
            }
        )
    source_evidence = []
    for file in ("rewards.py", "terminations.py"):
        source_path = f"src/active_perception_arms/tasks/insertion/mdp/{file}"
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"c179c89:{source_path}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            source_evidence.append(
                {
                    "revision": "c179c89",
                    "file": source_path,
                    "sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                }
            )
    return {
        "retained_configs_found": bool(saved),
        "saved_configs": saved,
        "source_evidence": source_evidence,
        "comparison": [
            {
                "aspect": "actor/critic",
                "historical": "Feedforward256,256,128",
                "current": "GRU128 then128,128; separate recurrent critic",
            },
            {"aspect": "Gaussian exploration std", "historical": 1.0, "current": 0.4},
            {
                "aspect": "initial adaptive LR / entropy",
                "historical": [0.001, 0.005],
                "current": [0.0003, 0.003],
            },
            {"aspect": "PPO epochs / minibatches", "historical": [5, 4], "current": [4, 8]},
            {"aspect": "configured total transitions", "historical": 49152000, "current": 18432000},
            {
                "aspect": "episode seconds / control dt",
                "historical": [5.0, 0.022],
                "current": [3.5, 0.04],
            },
            {"aspect": "policy RGB width,height", "historical": [96, 96], "current": [128, 96]},
            {
                "aspect": "reward",
                "historical": "-log(100*distance+1), replaced by1000 on instantaneous success (sourcec179c89)",
                "current": "3*potential change +10 on held success -time/action costs",
            },
            {
                "aspect": "success",
                "historical": "distance<2mm instantly",
                "current": "distance<2mm for three control samples",
            },
            {
                "aspect": "variant inertial properties",
                "historical": "Sourcec179c89 derives inertia from offset prong meshes; possible variant dynamics cue",
                "current": "Common mass/COM/inertia imposed explicitly across variants",
            },
        ],
        "interpretation": "Historical retained runs are an actual training reference, correcting the earlier unidentified-config assumption. Their training success summaries are not a matched held-out evaluation, and differing inertia, sensing, timing, budget, architecture and objective preclude attributing the gap to one factor.",
        "repair_recommendation": "Finish controlled optimization continuations first. If insufficient, separately test stronger distance shaping/success incentive while preserving corrected equal inertias and held-three success. Treat historical-style architecture/horizon/budget changes as labeled experiments, not changes to existing runs.",
        "heldout_evidence": "Missing independently evaluated historical checkpoints under the current equal-inertia held-three benchmark; no historical training-success number should be presented as matched validation/test success.",
    }


def instrumented_repeats(directory):
    rows = []
    for path in sorted(directory.glob("termination-*-repeat[12].json")):
        report = json.loads(path.read_text())
        rows.append(
            {
                "path": str(path),
                "condition": report["experiment"]["condition"],
                "success_state_sample": report["experiment"].get(
                    "success_state_sample", "derived_substep"
                ),
                "episodes": report["episodes"],
                "successes": report["successes"],
                "per_variant": report["per_variant"],
                "termination_summary": {
                    k: v for k, v in report["termination_audit"].items() if k != "records"
                },
                "repeat_comparison": {
                    k: v
                    for k, v in report["repeat_comparison"].items()
                    if k != "changed_episode_indices"
                },
            }
        )
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("artifacts/cluster_continuation/manifest.json")
    )
    p.add_argument(
        "--output", type=Path, default=Path("artifacts/plug_analysis/training_audit.json")
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Read current run state/summary through authenticated API",
    )
    p.add_argument("--legacy-root", type=Path, default=Path("../active_perception_arms"))
    args = p.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    inventory = json.loads(args.manifest.with_name("run_inventory.json").read_text())
    api = None
    if args.wandb:
        import wandb

        api = wandb.Api(timeout=30)
    audits = []
    for row in manifest:
        directory = Path(row["directory"])
        cfg = json.loads((directory / "experiment.json").read_text())
        runner = json.loads((directory / "runner.json").read_text())
        history_path = directory / "iterations.jsonl"
        contents = history_path.read_text()
        if contents and not contents.endswith("\n"):
            raise ValueError(f"Finalized history has unterminated record: {history_path}")
        history = [json.loads(s) for s in contents.splitlines() if s.strip()]
        audit = audit_history(
            history, row["total_iterations"], cfg["num_envs"] * runner["num_steps_per_env"]
        )
        checkpoint = Path(row["expected_final_checkpoint"])
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        std = weights["actor_state_dict"]["distribution.std_param"].tolist()
        original_std = torch.load(directory / "model_99.pt", map_location="cpu", weights_only=True)[
            "actor_state_dict"
        ]["distribution.std_param"].tolist()
        wandb_meta = json.loads((directory / "wandb_run.json").read_text())
        wandb_readback = {"verified_online": False, "url": wandb_meta["url"]}
        if api is not None:
            try:
                run = api.run(f"{wandb_meta['entity']}/{wandb_meta['project']}/{wandb_meta['id']}")
                wandb_readback.update(
                    verified_online=True,
                    state=run.state,
                    last_history_step_index=run.history_keys.get("lastStep")
                    if hasattr(run, "history_keys")
                    else None,
                    summary={
                        k: v
                        for k, v in dict(run.summary).items()
                        if k.startswith(("train/", "runtime/", "completed_", "total_"))
                    },
                )
            except Exception as exc:
                # Do not expose authentication material from exception text.
                wandb_readback["readback_error_type"] = type(exc).__name__
        job = next(j for j in inventory["jobs"] if j["JobID"] == row["training_array_task_id"])
        evaluation = compact_evaluation(Path(row["evaluation_output"]))
        early_eval = compact_evaluation(
            Path(row["original_summary"]).with_name(f"evaluation-{row['condition']}-s0.json")
        )
        interim_path = args.manifest.parent / f"interim-evaluation-{row['condition']}-s0-i750.json"
        captures = json.loads(
            Path(evaluation["path"])
            .with_name(Path(evaluation["path"]).stem + "-traces")
            .joinpath("capture.json")
            .read_text()
        )
        timing = []
        for record in captures["records"]:
            if record["expected_success"]:
                with np.load(record["trace"]) as trace:
                    timing.append(
                        {
                            "episode": record["episode"],
                            "variant": record["variant"],
                            "last_pre_action_error_m": float(trace["error"][-1]),
                            "min_pre_action_error_m": float(trace["error"].min()),
                            "pre_action_error_above_2mm": bool(trace["error"][-1] >= 0.002),
                        }
                    )
        integrity = (
            audit["contiguous_unique_updates"]
            and audit["transition_accounting_correct"]
            and weights["iter"] == row["total_iterations"] - 1
            and job["State"] == "COMPLETED"
            and job["ExitCode"] == "0:0"
            and audit["nonfinite_logged_metrics"] == 0
        )
        audits.append(
            {
                "condition": row["condition"],
                "training_seed": cfg["seed"],
                "directory": str(directory),
                "source_revision": row["source_revision"],
                "slurm": {
                    k: job[k]
                    for k in ("JobID", "State", "ExitCode", "Start", "End", "Elapsed", "NodeList")
                },
                "integrity_passed": integrity,
                "checkpoint": {
                    "path": str(checkpoint),
                    "saved_iteration": weights["iter"],
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                },
                **audit,
                "action_std": {
                    "axis_labels": ["tcp_x", "tcp_y", "tcp_z"]
                    + (
                        ["camera_x", "camera_y", "camera_z", "camera_yaw", "camera_pitch"]
                        if len(std) > 3
                        else []
                    ),
                    "at_update_100": original_std,
                    "at_update_1500": std,
                    "manipulation_mean_final": float(np.mean(std[:3])),
                    "camera_mean_final": float(np.mean(std[3:])) if len(std) > 3 else None,
                    "manipulation_axes_below_0_01_final": sum(v < 0.01 for v in std[:3]),
                    "note": "Normalized action-space standard deviations; cross-condition aggregate means obscure low manipulation exploration under high camera std. Low std is diagnostic, not proof of cause.",
                },
                "evaluations": {
                    "update_100": early_eval,
                    "update_751": compact_evaluation(interim_path)
                    if interim_path.exists()
                    else None,
                    "update_1500": evaluation,
                },
                "selected_success_capture_timing": timing,
                "wandb": wandb_readback,
                "task_solved_reliably": False,
                "task_quality_note": "All conditions have 0/128 xm validation successes; final overall success <0.37. Reward gains or plateaus do not establish reliable insertion.",
            }
        )
    output = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inventory_checked_at": inventory["checked_at"],
        "runs": audits,
        "historical_saved_config_comparison": historical_comparison(args.legacy_root),
        "measurement_audit": {
            "instrumented_repeats": instrumented_repeats(args.output.parent),
            "success_semantics": "Plug object-origin distance to offset-corrected socket goal <2mm on three consecutive control samples; not explicit prong contact/depth evidence.",
            "derived_state_lag": "Pinned MjLab1.4 step() computes terminations/rewards before sim.forward(); derived object/site positions lag qpos by one 2ms physics substep. Captured policy inputs/errors are post-forward pre-action samples, not exact termination samples.",
            "timing_conclusion": "Instrumented GPU repeats verify original derived-state hold-three flags, yet28/181 and40/187 initial,30/151 and31/140 active flagged successes have current-qpos errors>=2mm. This is a material measurement mismatch; original scores remain labeled legacy sampling. Corrected evaluation must measure current physical state for three steps, not subtract those counts retroactively.",
            "historical_repeat_audit": json.loads(
                Path("artifacts/cluster_pilot/evaluation_repeat_audit.json").read_text()
            ),
            "reproducibility_limit": "Two current initial/active repeats start from identical combined initial-physics-and-actor-image hashes in every512 episode, yet86 initial and125 active outcomes change. Reset mismatch is not the explanation for these measured repeats; numerical/physics/render trajectory divergence remains unresolved. Use repeated evaluations and training-seed replication; do not claim bitwise GPU determinism or unqualified per-episode causal pairing.",
            "implemented_repair": "Fresh experiments now default to current_qpos success sampling; saved old configs missing this field retain derived_substep. Resume inherits saved sampling and rejects explicit changes. Evaluator supports an explicit sampling intervention; its audit reports chosen manager samples and separate current/derived errors. Split physics-state and per-sensor image hashes are available for future localization.",
            "uncertainty_limit": "Wilson intervals describe episode sampling conditional on one training seed; they exclude training-seed uncertainty. Available evaluations are validation, not untouched test results.",
        },
        "decision": {
            "training_execution_complete": all(x["integrity_passed"] for x in audits),
            "reliable_task_learning": False,
            "active_perception_benefit": "No demonstrated benefit of continued active movement over initial inspection with memory. Initial 185/512 vs active141/512 is exploratory single-seed evidence; searched fixed view and causal interventions are incomplete.",
            "next_step": "Continue existing fixed_lr/entropy/both candidates to matched9,228,288 total transitions under their frozen original sampling. Evaluate all four final policies using corrected current-qpos held-three sampling, then test a separate stronger reward profile if controlled tuning remains insufficient. Gate new-task training and full condition replication on reliable task learning.",
            "unresolved_cause": "Learning-rate floor and per-axis std collapse are consistent with optimization trouble, but xm0 may also reflect sensing/alignment bias. Scripted feasibility and action-bound algebra reject a universal geometric impossibility; sampled policy failures do not establish causation.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for x in audits:
        print(
            x["condition"],
            "integrity",
            x["integrity_passed"],
            "reward_gain",
            round(x["reward_early_to_late_gain"], 3),
            "plateau",
            x["reward_plateau"]["descriptive_plateau"],
            "validation",
            x["evaluations"]["update_1500"]["successes"],
        )


if __name__ == "__main__":
    main()
