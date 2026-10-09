"""Prepare five fixed original-checkpoint camera controls, twice, on one GPU."""

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from submit_reward_continuation import (
    BOOTSTRAP_FIX,
    SOURCE_FILES,
    clean_environment,
    git,
    read,
    require,
    sha256,
    submit_reserved,
    validate_source,
)

CASES = (
    ("nominal-active", "active", []),
    ("freeze1-active", "active", ["--freeze-camera-after", "1.0"]),
    ("nominal-initial", "initial", []),
    ("hold1-initial", "initial", ["--hold-external-after", "1.0"]),
    ("resetmemory-initial", "initial", ["--reset-memory"]),
)
CONTROL_SOURCE_FILES = (
    *SOURCE_FILES,
    "src/active_perception_arms/evaluation_audit.py",
    "src/active_perception_arms/occlusion.py",
    "src/active_perception_arms/scenes.py",
    "src/active_perception_arms/plug.py",
    "src/active_perception_arms/robots/plug_control.py",
)
# Extend the existing observer only. The manager's compute remains first, and
# no physics, forward, learner or random-number call is added.
EVALUATE_WRAPPER = """
import sys
from active_perception_arms import evaluation_audit, mdp

class PositionAudit(evaluation_audit.TerminationAudit):
    def compute(self):
        done = super().compute()
        _, _, _, goal = mdp.positions(self.env)
        address = self.object_qpos_address
        body = self.env.sim.data.qpos[:, address:address + 3]
        self.snapshot.update(
            body_qpos_position_world_m=body.clone(),
            body_goal_world_m=goal.clone(),
            body_qpos_error_world_m=(body - goal).clone(),
        )
        return done

evaluation_audit.TerminationAudit = PositionAudit
from active_perception_arms.evaluate import main
main(sys.argv[1:])
"""
VIOLATIONS = (
    "success_hold_violations",
    "success_distance_violations",
    "success_three_sample_violations",
    "success_qpos_above_distance_threshold",
)
LIMITATIONS = [
    "Within-checkpoint interventions diagnose dependence of these trained policies; they do not establish a generally optimal sensing strategy.",
    "Camera freeze holds joint/gimbal command targets from 1s; actual joints can settle. RGB stays live. The intervention also changes camera embodiment and possible contacts.",
    "Initial-only already fixes camera targets after 1s. Holding external RGB after 1s retains wrist RGB, proprioception and GRU state; it tests continued external feedback at that trained viewpoint.",
    "resetmemory-initial resets the entire actor GRU every step, including inspection. It can change initial view selection and manipulation; it is not a camera-specific memory ablation or a retrained feedforward baseline.",
    "Matching initial physics and sensor hashes does not guarantee identical GPU contact trajectories. Outcome flips are descriptive; episodes are not IID causal intervention replicates.",
    "Two GPU execution repeats of training seed 0 quantify numerical sensitivity, not variation across independently trained policies. Wilson intervals are episode-count summaries, not causal-effect or training-seed uncertainty.",
    "Original actors were trained with derived_substep success and without the critic-bootstrap fix. Evaluation overrides current_qpos; no parameters are retrained. xm failure and unstable manipulation remain unresolved.",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def compare_initial_and_outcomes(reference, report):
    require(len(reference["outcomes"]) == len(report["outcomes"]) == 512, "Episode counts differ")
    physical = sum(
        a != b
        for a, b in zip(
            reference["initial_physics_state_sha256"],
            report["initial_physics_state_sha256"],
            strict=True,
        )
    )
    state = sum(
        a != b
        for a, b in zip(
            reference["initial_state_sha256"], report["initial_state_sha256"], strict=True
        )
    )
    require(
        set(reference["initial_sensor_sha256"])
        == set(report["initial_sensor_sha256"])
        == {"wrist", "external"},
        "Expected both actor cameras",
    )
    images = {
        name: sum(
            a != b
            for a, b in zip(
                reference["initial_sensor_sha256"][name],
                report["initial_sensor_sha256"][name],
                strict=True,
            )
        )
        for name in reference["initial_sensor_sha256"]
    }
    require(
        physical == state == 0 and all(value == 0 for value in images.values()),
        "Within-checkpoint initial physics/images do not match",
    )
    require(
        reference["checkpoint"] == report["checkpoint"],
        "Initial pairing must remain within one checkpoint",
    )
    require(reference["variants"] == report["variants"], "Variant assignment differs")
    lost = [
        index
        for index, (before, after) in enumerate(
            zip(reference["outcomes"], report["outcomes"], strict=True)
        )
        if before and not after
    ]
    gained = [
        index
        for index, (before, after) in enumerate(
            zip(reference["outcomes"], report["outcomes"], strict=True)
        )
        if after and not before
    ]
    return {
        "initial_state_mismatches": state,
        "initial_physics_mismatches": physical,
        "initial_sensor_mismatches": images,
        "changed_episode_count": len(lost) + len(gained),
        "changed_episode_fraction": (len(lost) + len(gained)) / 512,
        "lost_successes": len(lost),
        "gained_successes": len(gained),
        "lost_episode_indices": lost,
        "gained_episode_indices": gained,
        "success_count_change": report["successes"] - reference["successes"],
        "descriptive_only": True,
        "note": "Numerical-repeat/intervention flips, not episode-IID causal evidence.",
    }


def report_summary(report):
    from active_perception_arms.evaluate import wilson

    audit = report["termination_audit"]
    require(
        report["episodes"] == audit["episodes"] == 512
        and report["experiment"]["num_envs"] == 128
        and report["seed"] == 10000
        and report["split"] == "validation"
        and report["training_seed"] == 0
        and report["experiment"]["success_state_sample"] == "current_qpos"
        and set(report["per_variant"]) == {"xm", "xp", "ym", "yp"}
        and all(row["episodes"] == 128 for row in report["per_variant"].values()),
        "Physical protocol differs",
    )
    require(all(audit[key] == 0 for key in VIOLATIONS), "Current-qpos physical audit failed")
    require(
        all(
            len(report[key]) == 512
            for key in (
                "initial_state_sha256",
                "initial_physics_state_sha256",
                "outcomes",
                "variants",
            )
        ),
        "Incomplete paired audit",
    )
    records = audit["records"]
    require(
        len(records) == 512 and sorted(row["episode"] for row in records) == list(range(512)),
        "Incomplete terminal positions",
    )
    for row in records:
        vector = row["body_qpos_error_world_m"]
        require(
            len(vector) == 3 and all(math.isfinite(v) for v in vector),
            "Nonfinite terminal error vector",
        )
        require(
            abs(math.sqrt(sum(v * v for v in vector)) - row["qpos_position_error_m"]) < 1e-7,
            "Terminal vector/scalar error differs",
        )
    per_variant = {}
    for variant, counts in report["per_variant"].items():
        rows = [row for row in records if report["variants"][row["episode"]] == variant]
        per_variant[variant] = {
            **counts,
            "wilson_95": wilson(counts["successes"], counts["episodes"]),
            "mean_terminal_error_xyz_m": [
                sum(row["body_qpos_error_world_m"][i] for row in rows) / len(rows) for i in range(3)
            ],
            "mean_absolute_terminal_error_xyz_m": [
                sum(abs(row["body_qpos_error_world_m"][i]) for row in rows) / len(rows)
                for i in range(3)
            ],
        }
    return {
        key: report[key]
        for key in (
            "successes",
            "episodes",
            "success_rate",
            "wilson_95",
            "mean_camera_joint_travel_rad",
            "mean_episode_seconds",
            "mean_success_completion_seconds",
            "evaluation_wall_seconds",
        )
    } | {
        "per_variant": per_variant,
        "physical_violation_counts": {key: audit[key] for key in VIOLATIONS},
        "mean_terminal_position_error_m": audit["mean_terminal_position_error_m"],
        "max_derived_to_qpos_displacement_m": audit["max_derived_to_qpos_displacement_m"],
    }


def validate_record(record):
    source = Path(record["source_root"])
    require(
        git(source, "rev-parse", "HEAD") == record["source_revision"] == BOOTSTRAP_FIX,
        "Frozen source changed",
    )
    require(
        not git(source, "status", "--porcelain", "--untracked-files=no"),
        "Frozen source has tracked changes",
    )
    for relative, expected in record["source_files_sha256"].items():
        require(sha256(source / relative) == expected, f"Source hash differs: {relative}")
    for path, expected in record["launcher_files_sha256"].items():
        require(sha256(Path(path)) == expected, f"Launcher hash differs: {path}")
    for checkpoint in record["checkpoints"].values():
        for path, expected in checkpoint["files_sha256"].items():
            require(sha256(Path(path)) == expected, f"Checkpoint/config hash differs: {path}")


def runtime(manifest_path):
    record = read(manifest_path)
    validate_record(record)
    require(
        record["status"] == "submitted" and record["job_id"] == os.environ["SLURM_JOB_ID"],
        "Allocation differs from reservation",
    )
    require(
        bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
        and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1,
        "Expected one GPU",
    )
    require(
        sha256(Path(os.environ["CAMERA_CONTROLS_SCRIPT"])) == record["script_sha256"],
        "Submitted Slurm script hash differs",
    )
    output, source = Path(record["output_directory"]), Path(record["source_root"])
    runtime_path = output / "runtime_manifest.json"
    journal = {"status": "running", "job_id": record["job_id"], "started_at": now(), "stages": []}
    with runtime_path.open("x") as stream:
        json.dump(journal, stream, indent=2)
    summary = {
        "status": "running",
        "source_revision": record["source_revision"],
        "checkpoints": record["checkpoints"],
        "protocol": record["protocol"],
        "limitations": LIMITATIONS,
        "training_started": False,
        "cases": {},
    }
    deadline = time.monotonic() + 1170  # Flush failure/status before the 20min Slurm cap.
    try:
        for name, condition, intervention in CASES:
            directory = output / name
            directory.mkdir(exist_ok=False)
            checkpoint = record["checkpoints"][condition]["checkpoint"]
            case = {
                "condition": condition,
                "intervention": intervention,
                "checkpoint": checkpoint,
                "repeats": [],
            }
            summary["cases"][name] = case
            for repeat in (1, 2):
                validate_record(record)
                report_path = directory / f"evaluation-repeat{repeat}.json"
                require(not report_path.exists(), "Evaluation output exists; refuse overwrite")
                nominal = output / f"nominal-{condition}" / f"evaluation-repeat{repeat}.json"
                reference = None
                if name != f"nominal-{condition}":
                    reference = nominal
                elif repeat == 2:
                    reference = directory / "evaluation-repeat1.json"
                command = [
                    str(source / ".venv/bin/python"),
                    "-c",
                    EVALUATE_WRAPPER,
                    checkpoint,
                    "--num-envs",
                    "128",
                    "--episodes",
                    "512",
                    "--seed",
                    "10000",
                    "--split",
                    "validation",
                    "--success-state-sample",
                    "current_qpos",
                    "--audit-termination",
                    "--output",
                    str(report_path),
                    *intervention,
                ]
                if reference is not None:
                    command += ["--reference-report", str(reference)]
                stage = {
                    "case": name,
                    "repeat": repeat,
                    "command": command,
                    "started_at": now(),
                    "reference": str(reference) if reference else None,
                }
                journal["stages"].append(stage)
                save(runtime_path, journal)
                print(json.dumps(stage), flush=True)
                remaining = deadline - time.monotonic()
                require(remaining > 0, "Bounded evaluation time exhausted")
                with (directory / f"evaluation-repeat{repeat}.log").open("x") as log:
                    result = subprocess.run(
                        ["srun", "--ntasks=1", "--chdir", str(source), *command],
                        cwd=source,
                        env={**clean_environment(), "PYTHONPATH": str(source / "src")},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=remaining,
                    )
                stage.update(returncode=result.returncode, finished_at=now())
                save(runtime_path, journal)
                require(result.returncode == 0, f"{name}/repeat{repeat} failed")
                report = read(report_path)
                require(
                    report["checkpoint"] == checkpoint
                    and report["experiment"]["condition"] == condition,
                    "Scored checkpoint/condition differs",
                )
                expected_interventions = {
                    "freeze_camera_after": 1.0 if name == "freeze1-active" else None,
                    "hold_external_after": 1.0 if name == "hold1-initial" else None,
                    "reset_memory": name == "resetmemory-initial",
                    "camera_trace_in": None,
                    "success_state_sample_override": "current_qpos",
                }
                require(
                    report["interventions"] == expected_interventions, "Scored intervention differs"
                )
                stats = report_summary(report)
                stats.update(
                    repeat=repeat, report=str(report_path), report_sha256=sha256(report_path)
                )
                if reference:
                    stats["paired_with_reference"] = compare_initial_and_outcomes(
                        read(reference), report
                    )
                case["repeats"].append(stats)
                save(output / "summary.json", summary)
            case["within_case_repeat_variability"] = compare_initial_and_outcomes(
                read(directory / "evaluation-repeat1.json"),
                read(directory / "evaluation-repeat2.json"),
            )
            save(output / "summary.json", summary)
        validate_record(record)
        summary.update(status="complete", finished_at=now())
        journal.update(status="complete", finished_at=now())
        save(output / "summary.json", summary)
        save(runtime_path, journal)
        print(
            json.dumps(
                {"status": "complete", "summary": str(output / "summary.json"), "evaluations": 10}
            ),
            flush=True,
        )
    except BaseException as error:
        journal.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
        summary.update(status="failed", error=journal["error"])
        save(runtime_path, journal)
        save(output / "summary.json", summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--runtime", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runtime:
        runtime(args.runtime)
        return
    data = args.data_root.resolve()
    source = (args.source_root or data.parent / "plug-bootstrap-20261009").resolve()
    require(validate_source(source, data) == BOOTSTRAP_FIX, "Use the exact frozen 11a591f source")
    output = data / "artifacts/plug_analysis/camera_controls"
    require(
        not output.exists(), "Camera-control output already exists; inspect rather than overwrite"
    )
    checkpoints = {}
    import torch

    for condition in ("active", "initial"):
        original = read(data / f"artifacts/plug_analysis/corrected-{condition}-s0.json")
        checkpoint = Path(original["checkpoint"]).resolve()
        require(
            checkpoint.name == "model_1499.pt"
            and torch.load(checkpoint, map_location="cpu", weights_only=True)["iter"] == 1499,
            "Expected original final 1500-update checkpoint",
        )
        cfg = read(checkpoint.with_name("experiment.json"))
        require(
            cfg["task"] == "plug"
            and cfg["condition"] == condition
            and cfg["memory"] == "gru"
            and cfg["num_envs"] == 512
            and cfg["seed"] == 0
            and cfg["occlusion"] == "clean"
            and cfg["width"] == 128
            and cfg["height"] == 96
            and cfg["episode_seconds"] == 3.5
            and cfg["initial_seconds"] == 1.0
            and cfg["randomize"] is True
            and cfg["timestep"] == 0.002
            and cfg["decimation"] == 20
            and cfg["plug_variant"] is None,
            "Original trained configuration differs",
        )
        paths = (
            checkpoint,
            checkpoint.with_name("experiment.json"),
            checkpoint.with_name("runner.json"),
        )
        checkpoints[condition] = {
            "checkpoint": str(checkpoint),
            "files_sha256": {str(path): sha256(path) for path in paths},
            "trained_updates": 1500,
            "trained_transitions": 18432000,
            "training_success_state_sample": cfg.get("success_state_sample", "derived_substep"),
            "existing_current_qpos_successes": original["successes"],
        }
    helper = Path(__file__).resolve()
    script = data / "scripts/slurm/camera_controls.sbatch"
    manifest_path = output / "submission.json"
    settings = {
        "PROJECT_ROOT": str(source),
        "PYTHONPATH": str(source / "src"),
        "CAMERA_CONTROLS_MANIFEST": str(manifest_path),
        "CAMERA_CONTROLS_HELPER": str(helper),
    }
    command = [
        "sbatch",
        "--parsable",
        "--account=kempner_pgozdil_lab",
        "--partition=kempner_rtx",
        "--export=ALL",
        "--exclude=holygpu7c2313",
        f"--output={data}/logs/slurm/%x-%j.out",
        f"--error={data}/logs/slurm/%x-%j.err",
        str(script),
    ]
    record = {
        "source_root": str(source),
        "source_revision": BOOTSTRAP_FIX,
        "data_root": str(data),
        "output_directory": str(output),
        "source_files_sha256": {path: sha256(source / path) for path in CONTROL_SOURCE_FILES},
        "launcher_files_sha256": {
            str(path): sha256(path)
            for path in (
                helper,
                data / "scripts/submit_reward_continuation.py",
                data / "scripts/submit_tuning_continuation.py",
            )
        },
        "script_sha256": sha256(script),
        "checkpoints": checkpoints,
        "cases": [
            {"name": name, "condition": condition, "flags": flags}
            for name, condition, flags in CASES
        ],
        "protocol": {
            "episodes_per_case_repeat": 512,
            "episodes_per_variant": 128,
            "num_envs": 128,
            "seed": 10000,
            "split": "validation",
            "success_state_sample": "current_qpos",
            "audit_termination": True,
            "reset_seeds": [10000, 10001, 10002, 10003],
            "terminal_position_snapshot": "Raw object freejoint XYZ and offset-corrected goal XYZ in world meters, before automatic reset; read-only extension of the unchanged termination observer.",
            "repeats": 2,
            "evaluations": 10,
            "total_evaluated_episodes": 5120,
            "gpus": 1,
            "slurm_time_minutes": 20,
            "wandb": False,
            "videos": False,
        },
        "limitations": LIMITATIONS,
        "prepared_at": now(),
        "command": command,
        "environment": settings,
        "training_started": False,
    }
    if not args.submit:
        print(json.dumps(record, indent=2))
        return
    env = clean_environment()
    env.update(settings)
    record = submit_reserved(manifest_path, record, command, source, env)
    print(
        json.dumps(
            {"job_id": record["job_id"], "evaluations": 10, "gpus": 1, "time_minutes": 20}, indent=2
        )
    )


if __name__ == "__main__":
    main()
