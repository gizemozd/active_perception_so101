"""Prepare one fresh exploration repair; objective selection and submission are explicit."""

import argparse
import json
import math
import os
import secrets
import shlex
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from submit_reward_continuation import (
    BOOTSTRAP_FIX,
    PROFILES,
    SOURCE_FILES,
    clean_environment,
    git,
    read,
    require,
    sha256,
    submit_reserved,
    validate_source,
)

UPDATES = 751
TRANSITIONS = 9228288
VIOLATIONS = (
    "success_hold_violations",
    "success_distance_violations",
    "success_three_sample_violations",
    "success_qpos_above_distance_threshold",
)
CODE_CHECK = r"""
import hashlib, json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
for path, expected in record['launcher_files_sha256'].items():
    with Path(path).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, path
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def physical_report(report, profile):
    require(
        report["episodes"] == 512
        and report["seed"] == 10000
        and report["split"] == "validation"
        and report["training_seed"] == 0,
        "Expected the balanced seed10000 validation protocol",
    )
    require(
        report["experiment"]["success_state_sample"] == "current_qpos"
        and report["experiment"]["reward_profile"] == profile
        and report["experiment"]["num_envs"] == 128
        and report["experiment"]["task"] == "plug"
        and report["experiment"]["condition"] == "wrist_static"
        and report["experiment"]["occlusion"] == "clean"
        and report["experiment"]["memory"] == "gru",
        "Physical evaluation configuration differs",
    )
    require(
        set(report["per_variant"]) == {"xm", "xp", "ym", "yp"}
        and all(row["episodes"] == 128 for row in report["per_variant"].values()),
        "Expected 128 episodes for each plug variant",
    )
    audit = report["termination_audit"]
    require(audit["episodes"] == 512, "Incomplete physical audit")
    require(
        len(report["outcomes"]) == len(report["variants"]) == 512
        and sum(report["outcomes"]) == report["successes"]
        and audit["terminal_flags"]["success"] == report["successes"]
        and all(
            row["successes"]
            == sum(
                outcome
                for outcome, observed in zip(report["outcomes"], report["variants"], strict=True)
                if observed == variant
            )
            for variant, row in report["per_variant"].items()
        ),
        "Variant counts, episode outcomes and physical audit disagree",
    )
    return {
        "minimum_successes_per_variant": 103,
        "episodes_per_variant": 128,
        "per_variant_successes": {
            variant: row["successes"] for variant, row in report["per_variant"].items()
        },
        "zero_physical_violations": all(audit[key] == 0 for key in VIOLATIONS),
        "physical_violation_counts": {key: audit[key] for key in VIOLATIONS},
        "passed": all(row["successes"] >= 103 for row in report["per_variant"].values())
        and all(audit[key] == 0 for key in VIOLATIONS),
    }


def entry_evidence(data):
    evidence = {}
    for profile in PROFILES:
        root = data / "artifacts/plug_analysis/reward_repair/continuation" / profile
        manifest = read(root / "run_manifest.json")
        require(
            manifest["status"] == "complete"
            and manifest["training_updates"] == UPDATES
            and manifest["training_transitions"] == TRANSITIONS,
            "Both corrected751 runs must finish first",
        )
        checkpoint = Path(manifest["checkpoint"])
        require(
            checkpoint.name == "model_750.pt"
            and sha256(checkpoint) == manifest["checkpoint_sha256"],
            "Corrected final checkpoint identity differs",
        )
        reports = [read(root / f"evaluation-repeat{repeat}.json") for repeat in (1, 2)]
        require(
            all(report["checkpoint"] == str(checkpoint) for report in reports),
            "Entry evidence is not the final checkpoint",
        )
        gates = [physical_report(report, profile) for report in reports]
        require(
            all(gate["zero_physical_violations"] for gate in gates),
            "Repair physical audit errors before exploring",
        )
        evidence[profile] = {
            "manifest": str(root / "run_manifest.json"),
            "manifest_sha256": sha256(root / "run_manifest.json"),
            "repeat_gates": gates,
            "competent_each_repeat": all(gate["passed"] for gate in gates),
        }
    require(
        not any(record["competent_each_repeat"] for record in evidence.values()),
        "A corrected751 profile meets the competence gate; exploration repair is not the approved fallback",
    )
    return evidence


def validate_record(record):
    source = Path(record["source_root"])
    require(
        git(source, "rev-parse", "HEAD") == record["source_revision"] == BOOTSTRAP_FIX,
        "Exploration training source must remain11a591f",
    )
    require(
        not git(source, "status", "--porcelain", "--untracked-files=no"),
        "Frozen source has tracked changes",
    )
    for relative, expected in record["source_files_sha256"].items():
        require(sha256(source / relative) == expected, f"Frozen source changed: {relative}")
    for path, expected in record["launcher_files_sha256"].items():
        require(sha256(Path(path)) == expected, f"Launcher dependency changed: {path}")


def identify_run(record):
    log_root = Path(record["log_root"])
    if not log_root.exists():
        return None
    prefix = f"plug_wrist_static_clean_gru_v7_s0_n512_{record['run_label']}_"
    candidates = [
        path for path in log_root.iterdir() if path.is_dir() and path.name.startswith(prefix)
    ]
    require(
        len(candidates) <= 1, "Ambiguous exploration run identity; no latest-directory selection"
    )
    if not candidates or not all(
        (candidates[0] / name).exists()
        for name in ("runtime.json", "experiment.json", "wandb_run.json")
    ):
        return None
    directory = candidates[0]
    try:
        runtime = read(directory / "runtime.json")
        experiment = read(directory / "experiment.json")
        wandb = read(directory / "wandb_run.json")
    except (json.JSONDecodeError, FileNotFoundError):
        return None  # Metadata files can exist before their first write finishes.
    require(
        runtime["slurm_job_id"] == record["job_id"]
        and runtime["git_revision"] == record["source_revision"]
        and runtime["resume"] is None,
        "Fresh run source/job/resume identity differs",
    )
    require(
        experiment == record["dry_run"]["experiment"],
        "Fresh saved experiment differs",
    )
    require(
        wandb["id"] == record["training_wandb_id"],
        "Fresh training W&B identity differs",
    )
    return directory


def rows_at(directory):
    path = directory / "iterations.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # The logger may be writing the final line while observed.
    return rows


def diagnostics(output, rows, threshold, filename):
    path = output / filename
    if len(rows) < threshold or path.exists():
        return
    selected = rows[:threshold]
    keys = [key for key in selected[-1] if key.startswith(("train/Loss/", "train/Policy/"))]
    nonfinite = [
        {"iteration": row["iteration"], "key": key}
        for row in selected
        for key in keys
        if isinstance(row.get(key), (int, float)) and not math.isfinite(row[key])
    ]
    completed = sum(row.get("train/Loss/diagnostic_completed_episodes", 0) for row in selected)
    successes = sum(row.get("train/Loss/diagnostic_successes", 0) for row in selected)
    save(
        path,
        {
            "read_only": True,
            "hyperparameters_changed": False,
            "training_updates_observed": threshold,
            "training_transitions": threshold * 12288,
            "observed_at": now(),
            "completed_episodes": completed,
            "successes": successes,
            "episode_weighted_success": successes / completed if completed else None,
            "nonfinite_training_metrics": nonfinite,
            "last_metrics": {key: selected[-1][key] for key in keys},
            "last10_mean_metrics": {
                key: statistics.fmean(row[key] for row in selected[-10:] if key in row)
                for key in keys
            },
            "note": "Telemetry only; no checkpoint selection, early extension, or exploration setting changes.",
        },
    )


def runtime_training(manifest_path):
    record = read(manifest_path)
    validate_record(record)
    require(
        record["status"] == "submitted" and os.environ["SLURM_JOB_ID"] == record["job_id"],
        "Training allocation differs from its reservation",
    )
    require(
        bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
        and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1,
        "Expected one visible training GPU",
    )
    output = Path(record["output_directory"])
    log_root = Path(record["log_root"])
    require(
        not log_root.exists() or not any(log_root.iterdir()), "Fresh profile log root is not empty"
    )
    require(not (output / "training.json").exists(), "Fresh training summary already exists")
    runtime = {
        "status": "running",
        "started_at": now(),
        "job_id": record["job_id"],
        "source_revision": record["source_revision"],
        "training_command": record["training_command"],
    }
    with (output / "training-runtime.json").open("x") as stream:
        json.dump(runtime, stream, indent=2)
    process = subprocess.Popen(
        ["srun", "--ntasks=1", "--chdir", record["source_root"], *record["training_command"]],
        cwd=record["source_root"],
    )
    try:
        while process.poll() is None:
            directory = identify_run(record)
            if directory is not None:
                rows = rows_at(directory)
                diagnostics(output, rows, 100, "diagnostics-100.json")
                diagnostics(output, rows, 376, "diagnostics-midpoint.json")
            time.sleep(10)
        require(process.returncode == 0, f"Training failed with exit code {process.returncode}")
        validate_record(record)
        directory = identify_run(record)
        require(directory is not None, "Fresh labeled run was not created")
        rows = rows_at(directory)
        require(
            [row["iteration"] for row in rows] == list(range(1, 752)),
            "Expected751 complete training rows",
        )
        require(
            all(
                row["slurm_job_id"] == record["job_id"]
                and row["transitions"] == row["iteration"] * 12288
                for row in rows
            ),
            "Training identity/budget differs",
        )
        training = read(output / "training.json")
        require(
            training["actual_transitions"]
            == training["final_cumulative_transitions"]
            == TRANSITIONS,
            "Fresh751 budget incomplete",
        )
        runner = read(directory / "runner.json")
        require(
            runner["num_steps_per_env"] == 24
            and runner["max_iterations"] == UPDATES
            and runner["actor"]["distribution_cfg"]["init_std"] == 1.0
            and {key: runner["algorithm"][key] for key in record["optimization"]}
            == record["optimization"],
            "Fresh exploration settings differ",
        )
        import torch

        checkpoint = directory / "model_750.pt"
        require(
            torch.load(checkpoint, map_location="cpu", weights_only=True)["iter"] == 750,
            "Final checkpoint is notmodel750",
        )
        diagnostics(output, rows, 100, "diagnostics-100.json")
        diagnostics(output, rows, 376, "diagnostics-midpoint.json")
        completed = {
            **record,
            "status": "trained",
            "run_directory": str(directory),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "training_runtime": read(directory / "runtime.json"),
            "training_wandb": read(directory / "wandb_run.json"),
            "evaluations": [],
        }
        save(output / "run_manifest.json", completed)
        runtime.update(status="complete", finished_at=now())
        save(output / "training-runtime.json", runtime)
    except BaseException as error:
        if process.poll() is None:
            process.terminate()
            process.wait()
        runtime.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
        save(output / "training-runtime.json", runtime)
        raise


def runtime_evaluation(manifest_path):
    record = read(manifest_path)
    validate_record(record)
    output = Path(record["output_directory"])
    evaluation = read(output / "evaluation-submission.json")
    require(
        evaluation["status"] == "submitted" and evaluation["job_id"] == os.environ["SLURM_JOB_ID"],
        "Evaluation reservation differs",
    )
    require(
        bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
        and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1,
        "Expected one visible evaluation GPU",
    )
    require(
        sha256(Path(os.environ["EXPLORATION_EVAL_SCRIPT"])) == evaluation["script_sha256"],
        "Evaluation script changed",
    )
    manifest = read(output / "run_manifest.json")
    require(
        manifest["status"] == "trained", "Afterok evaluation requires verified fresh751 training"
    )
    checkpoint = Path(manifest["checkpoint"])
    manifest.update(status="evaluating", evaluation_job_id=evaluation["job_id"])
    save(output / "run_manifest.json", manifest)
    runtime = {
        "status": "running",
        "job_id": evaluation["job_id"],
        "started_at": now(),
        "stages": [],
    }
    with (output / "evaluation-runtime.json").open("x") as stream:
        json.dump(runtime, stream, indent=2)

    def run(command, repeat, stage):
        validate_record(record)
        env = clean_environment()
        env.update(
            evaluation["environment"],
            WANDB_RUN_ID=evaluation["environment"][f"EVALUATION_WANDB_ID_{repeat}"],
            WANDB_RESUME="never",
        )
        started = {
            "stage": stage,
            "repeat": repeat,
            "started_at": now(),
            "command": command,
            "wandb_id": env["WANDB_RUN_ID"],
        }
        runtime["stages"].append(started)
        save(output / "evaluation-runtime.json", runtime)
        subprocess.run(
            ["srun", "--ntasks=1", "--chdir", record["source_root"], *command],
            cwd=record["source_root"],
            env=env,
            check=True,
        )
        started["finished_at"] = now()
        save(output / "evaluation-runtime.json", runtime)

    try:
        gates = []
        python = str(Path(record["source_root"]) / ".venv/bin/python")
        for repeat in (1, 2):
            report_path = output / f"evaluation-repeat{repeat}.json"
            require(not report_path.exists(), "Evaluation output already exists")
            require(sha256(checkpoint) == manifest["checkpoint_sha256"], "Final checkpoint changed")
            command = [
                python,
                "-m",
                "active_perception_arms.evaluate",
                str(checkpoint),
                "--episodes",
                "512",
                "--num-envs",
                "128",
                "--seed",
                "10000",
                "--split",
                "validation",
                "--success-state-sample",
                "current_qpos",
                "--audit-termination",
                "--output",
                str(report_path),
                "--wandb",
            ]
            if repeat == 1:
                command += ["--capture-representatives", str(output / "evaluation-repeat1-traces")]
            else:
                command += ["--reference-report", str(output / "evaluation-repeat1.json")]
            run(command, repeat, "validation")
            report = read(report_path)
            require(
                report["checkpoint"] == str(checkpoint)
                and report["wandb_id"]
                == evaluation["environment"][f"EVALUATION_WANDB_ID_{repeat}"],
                "Evaluated checkpoint or W&B identity differs",
            )
            gate = physical_report(report, record["profile"])
            gates.append(gate)
            item = {
                "repeat": repeat,
                "path": str(report_path),
                "sha256": sha256(report_path),
                "wandb_id": report["wandb_id"],
                "wandb_url": report["wandb_url"],
                "successes": report["successes"],
                "episodes": 512,
                "competence_gate": gate,
            }
            manifest["evaluations"].append(item)
            if repeat == 1:
                require(
                    bool(report["representative_capture"]), "Actual evaluated trajectories missing"
                )
                media = output / "evaluation-repeat1-videos"
                run(
                    [
                        python,
                        "-m",
                        "active_perception_arms.record_evaluation",
                        str(report_path),
                        "--output",
                        str(media),
                        "--wandb",
                    ],
                    repeat,
                    "captured-videos",
                )
                videos = read(media / "representatives.json")["records"]
                require(
                    videos
                    and all(
                        Path(row[key]).is_file() and Path(row[key]).stat().st_size > 0
                        for row in videos
                        for key in ("outside_video", "policy_video")
                    ),
                    "Captured videos missing",
                )
                item.update(
                    media_manifest=str(media / "representatives.json"),
                    media_manifest_sha256=sha256(media / "representatives.json"),
                )
            else:
                comparison = report["repeat_comparison"]
                same = (
                    comparison["initial_hash_comparable"]
                    and comparison["initial_hash_mismatches"] == 0
                    and comparison["initial_physics_state_mismatches"] == 0
                    and comparison["same_success_state_sample"]
                    and set(comparison["initial_sensor_image_mismatches"]) == {"wrist", "external"}
                    and all(
                        value == 0
                        for value in comparison["initial_sensor_image_mismatches"].values()
                    )
                )
                require(same, "Repeat initial physical/sensor hashes do not match")
                manifest["repeat_initial_hashes_match"] = True
            save(output / "run_manifest.json", manifest)
        manifest.update(
            status="complete",
            competence_passed=all(gate["passed"] for gate in gates),
            competence_gate={
                "required_successes_each_variant_each_repeat": 103,
                "episodes_per_variant": 128,
                "required_zero_physical_violations": True,
                "repeats": gates,
                "note": "Single training seed; this competence screen is not paper-level evidence.",
            },
            finished_at=now(),
        )
        save(output / "run_manifest.json", manifest)
        require(
            all(gate["zero_physical_violations"] for gate in gates),
            "Physical audit violations require diagnosis",
        )
        runtime.update(status="complete", finished_at=now())
        save(output / "evaluation-runtime.json", runtime)
    except BaseException as error:
        runtime.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
        manifest.update(status="failed", error=runtime["error"])
        save(output / "evaluation-runtime.json", runtime)
        save(output / "run_manifest.json", manifest)
        raise


def queue_evaluation(record, submit):
    output = Path(record["output_directory"])
    data, source = Path(record["data_root"]), Path(record["source_root"])
    script = data / "scripts/slurm/exploration_repair_eval.sbatch"
    env = clean_environment()
    settings = {
        **record["environment"],
        "EXPLORATION_MANIFEST": record["global_manifest"],
        "EXPLORATION_HELPER": str(Path(__file__).resolve()),
        "EVALUATION_WANDB_ID_1": secrets.token_hex(6),
        "EVALUATION_WANDB_ID_2": secrets.token_hex(6),
    }
    for key in ("WANDB_RUN_ID", "WANDB_RESUME"):
        settings.pop(key, None)
    env.update(settings)
    command = [
        "sbatch",
        "--parsable",
        "--account=kempner_pgozdil_lab",
        "--partition=kempner_rtx",
        "--export=ALL",
        "--exclude=holygpu7c2313",
        f"--dependency=afterok:{record.get('job_id', '<training-job>')}",
        f"--job-name=plug-explore-eval-{record['profile']}",
        f"--output={data}/logs/slurm/%x-%j.out",
        f"--error={data}/logs/slurm/%x-%j.err",
        str(script),
    ]
    evaluation = {
        "profile": record["profile"],
        "training_job_id": record.get("job_id"),
        "environment": settings,
        "command": command,
        "source_revision": record["source_revision"],
        "script_sha256": sha256(script),
    }
    return (
        submit_reserved(output / "evaluation-submission.json", evaluation, command, source, env)
        if submit
        else evaluation
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        choices=PROFILES,
        nargs="?",
        help="Required explicit objective; no automatic winner",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    internal = parser.add_mutually_exclusive_group()
    internal.add_argument("--runtime-training", type=Path, help=argparse.SUPPRESS)
    internal.add_argument("--runtime-evaluation", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runtime_training:
        runtime_training(args.runtime_training)
        return
    if args.runtime_evaluation:
        runtime_evaluation(args.runtime_evaluation)
        return
    if args.profile is None:
        parser.error("profile is required: choose progress or legacy_log_hold explicitly")
    data = args.data_root.resolve()
    source = (args.source_root or data.parent / "plug-bootstrap-20261009").resolve()
    revision = validate_source(source, data)
    require(revision == BOOTSTRAP_FIX, "Use the exact frozen11a591f source")
    root = data / "artifacts/plug_analysis/exploration_repair"
    global_manifest = root / "submission.json"
    if args.evaluation_only:
        record = read(global_manifest)
        require(
            record["status"] == "submitted"
            and record["profile"] == args.profile
            and record["source_revision"] == revision
            and record["source_root"] == str(source),
            "Existing reservation is not verified for this profile/source",
        )
        require(
            not (Path(record["output_directory"]) / "evaluation-submission.json").exists(),
            "Evaluation reservation exists; inspect accounting",
        )
        print(json.dumps(queue_evaluation(record, args.submit), indent=2))
        return
    require(
        not global_manifest.exists(),
        "One exploration repair is already reserved; inspect rather than duplicate",
    )
    output = root / args.profile
    log_root = data / "logs/exploration_repair" / args.profile
    require(
        not output.exists() and not log_root.exists(),
        "Fresh exploration output/log root already exists",
    )
    run_id = secrets.token_hex(6)
    run_label = f"explore_{args.profile}_{run_id}"
    command = [
        str(source / ".venv/bin/python"),
        "-m",
        "active_perception_arms.train",
        "--task",
        "plug",
        "--condition",
        "wrist_static",
        "--fixed-view",
        "7",
        "--num-envs",
        "512",
        "--seed",
        "0",
        "--occlusion",
        "clean",
        "--memory",
        "gru",
        "--success-state-sample",
        "current_qpos",
        "--reward-profile",
        args.profile,
        "--iterations",
        "751",
        "--initial-std",
        "1.0",
        "--entropy-coef",
        "0.01",
        "--learning-rate",
        "0.0003",
        "--lr-schedule",
        "adaptive",
        "--job-type",
        "search",
        "--run-label",
        run_label,
        "--log-root",
        str(log_root),
        "--result",
        str(output / "training.json"),
    ]
    env = clean_environment()
    settings = {
        "PROJECT_ROOT": str(source),
        "DATA_ROOT": str(data),
        "PYTHONPATH": str(source / "src"),
        "WANDB_RUN_ID": run_id,
        "WANDB_RESUME": "never",
        "WANDB_MODE": "online",
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "active-perception-so101"),
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", "pgozdil-harvard-university"),
        "WANDB_RUN_GROUP": f"plug-exploration-repair-{run_id}",
        "WANDB_JOB_TYPE": "search",
        "WANDB_TAGS": f"exploration-repair,fresh,initial-std1,entropy0.01,{args.profile}",
        "EVALUATION_STAGE": "fresh-exploration-repair",
    }
    env.update(settings)
    dry = json.loads(
        subprocess.check_output(command + ["--dry-run"], cwd=source, env=env, text=True)
    )
    expected = read(
        data / "artifacts/plug_analysis/reward_repair" / args.profile / "training.json"
    )["experiment"]
    require(
        dry["experiment"] == expected
        and dry["iterations"] == UPDATES
        and dry["initial_std"] == 1.0
        and dry["optimization"]
        == {"learning_rate": 0.0003, "schedule": "adaptive", "entropy_coef": 0.01}
        and not dry["training_started"],
        "Prepared exploration CLI/config differs",
    )
    helper = Path(__file__).resolve()
    dependencies = (
        helper,
        data / "scripts/submit_reward_continuation.py",
        data / "scripts/submit_tuning_continuation.py",
    )
    record = {
        "profile": args.profile,
        "source_root": str(source),
        "source_revision": revision,
        "data_root": str(data),
        "source_history": [{"revision": revision, "updates": UPDATES, "transitions": TRANSITIONS}],
        "source_files_sha256": {path: sha256(source / path) for path in SOURCE_FILES},
        "launcher_files_sha256": {str(path): sha256(path) for path in dependencies},
        "global_manifest": str(global_manifest),
        "output_directory": str(output),
        "log_root": str(log_root),
        "run_label": run_label,
        "training_command": command,
        "environment": settings,
        "training_wandb_id": run_id,
        "training_updates": UPDATES,
        "training_transitions": TRANSITIONS,
        "initial_std": 1.0,
        "optimization": dry["optimization"],
        "dry_run": dry,
        "fresh": True,
        "resume": None,
        "auto_extensions": False,
        "max_parallel_training_gpus": 1,
        "prepared_at": now(),
        "entry_condition": "Only if both final751 corrected profiles fail robust all-variant learning; caller selects objective after reviewing repeats.",
        "interpretation": "Fresh source11a751 updates with std1.0 and entropy0.01 changed together. One training seed is a competence screen, not paper-level proof.",
        "competence_gate": {
            "minimum_successes_each_variant_each_repeat": 103,
            "episodes_per_variant": 128,
            "repeats": 2,
            "zero_physical_violations_required": True,
        },
        "diagnostics": {"read_only_updates": [100, 376], "no_hyperparameter_changes": True},
    }
    wrap = "set -euo pipefail\nsource " + shlex.quote(str(source / "scripts/slurm/common.sh"))
    wrap += "\n" + shlex.join([command[0], "-c", CODE_CHECK, str(global_manifest)])
    wrap += "\nexec " + shlex.join(
        [command[0], str(helper), "--runtime-training", str(global_manifest)]
    )
    sbatch = [
        "sbatch",
        "--parsable",
        "--account=kempner_pgozdil_lab",
        "--partition=kempner_rtx",
        "--export=ALL",
        "--exclude=holygpu7c2313",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=48G",
        "--time=03:00:00",
        "--signal=USR1@180",
        f"--job-name=plug-explore-{args.profile}",
        f"--output={data}/logs/slurm/%x-%j.out",
        f"--error={data}/logs/slurm/%x-%j.err",
        "--wrap",
        "exec bash -c " + shlex.quote(wrap),
    ]
    record["command"] = sbatch
    evaluation = queue_evaluation(record, False)
    if not args.submit:
        print(
            json.dumps(
                {"training": record, "evaluation": evaluation, "training_started": False}, indent=2
            )
        )
        return
    record["entry_evidence"] = entry_evidence(data)
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    save(output / "run_manifest.json", {**record, "status": "prepared"})
    record = submit_reserved(global_manifest, record, sbatch, source, env)
    save(output / "submission.json", record)
    save(output / "run_manifest.json", {**record, "status": "training_submitted"})
    evaluation = queue_evaluation(record, True)
    record["evaluation_job_id"] = evaluation["job_id"]
    save(global_manifest, record)
    save(output / "submission.json", record)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "training_job": record["job_id"],
                "evaluation_job": evaluation["job_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
