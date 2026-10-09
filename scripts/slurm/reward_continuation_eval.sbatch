#!/usr/bin/env bash
# Submitted with afterok:<one-profile-training-job>; one GPU for both repeats.
# The caller records the screen/HP entry decision before submitting extensions.
#SBATCH --job-name=plug-reward751-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the frozen corrected source}"
: "${CONTINUATION_MANIFEST:?Set CONTINUATION_MANIFEST to the reserved training submission}"
export CONTINUATION_EVAL_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
source "${PROJECT_ROOT}/scripts/slurm/common.sh"
export PYTHONPATH="${PROJECT_ROOT}/src"
"${PROJECT_ROOT}/.venv/bin/python" - "$CONTINUATION_MANIFEST" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

submission_path = Path(sys.argv[1])
submission = json.loads(submission_path.read_text())
source = Path(submission["source_root"])
output = Path(submission["output_directory"])
profile = submission["profile"]
evaluation_submission = json.loads((output / "evaluation-submission.json").read_text())
python = str(source / ".venv/bin/python")
job = os.environ["SLURM_JOB_ID"]
runtime_path = output / "evaluation-runtime.json"
manifest_path = output / "run_manifest.json"
checkpoint = Path(submission["expected_final_checkpoint"])


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_source():
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    require(revision == submission["source_revision"], "Frozen source revision changed")
    require(
        not subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip(),
        "Frozen source has tracked changes",
    )
    for relative, expected in submission["source_files_sha256"].items():
        require(sha256(source / relative) == expected, f"Frozen source hash changed: {relative}")


require(
    evaluation_submission["status"] == "submitted" and evaluation_submission["job_id"] == job,
    "Evaluation allocation does not match its reservation",
)
require(
    sha256(Path(os.environ["CONTINUATION_EVAL_SCRIPT"])) == evaluation_submission["script_sha256"],
    "Evaluation script differs from submitted provenance",
)
require(
    bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
    and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1,
    "Expected one visible GPU",
)
validate_source()
runtime = {
    "status": "running",
    "slurm_job_id": job,
    "started_at": now(),
    "training_job_id": submission["job_id"],
    "source_revision": submission["source_revision"],
    "source_change": submission["source_change"],
    "hostname": platform.node(),
    "stages": [],
}
with runtime_path.open("x") as stream:
    json.dump(runtime, stream, indent=2)
manifest = read(manifest_path)


def run(command, repeat, stage):
    validate_source()
    env = os.environ.copy()
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_USERNAME", "WANDB_RUN_NAME"):
        env.pop(key, None)
    env.update(
        WANDB_RUN_ID=evaluation_submission["environment"][f"EVALUATION_WANDB_ID_{repeat}"],
        WANDB_RESUME="never",
        PYTHONPATH=str(source / "src"),
    )
    record = {
        "stage": stage,
        "repeat": repeat,
        "command": command,
        "cwd": str(source),
        "wandb_id": env["WANDB_RUN_ID"],
        "started_at": now(),
    }
    runtime["stages"].append(record)
    save(runtime_path, runtime)
    result = subprocess.run(
        ["srun", "--ntasks=1", "--chdir", str(source), *command], cwd=source, env=env
    )
    record.update(returncode=result.returncode, finished_at=now())
    save(runtime_path, runtime)
    require(result.returncode == 0, f"{stage} failed with exit code {result.returncode}")
    validate_source()


try:
    original = Path(submission["run_directory"])
    require(
        sha256(Path(submission["resume_checkpoint"])) == submission["resume_checkpoint_sha256"],
        "Original 100-update checkpoint changed",
    )
    require(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["iter"] == 750,
        "Expected the completed 751-update checkpoint",
    )
    checkpoint_hash = sha256(checkpoint)
    training = read(output / "training.json")
    train_runtime = read(original / "runtime.json")
    require(
        training["actual_transitions"] == 7999488
        and training["final_cumulative_transitions"] == 9228288,
        "Incomplete additional/cumulative training budget",
    )
    require(
        train_runtime["git_revision"] == submission["source_revision"]
        and train_runtime["slurm_job_id"] == submission["job_id"]
        and train_runtime["resume"] == submission["resume_checkpoint"]
        and train_runtime["resume_next_iteration_index"] == 100,
        "Training source, job or resume identity mismatch",
    )
    require(
        train_runtime["resume_learning_rate"] == submission["resume_optimizer_lr"],
        "Adaptive optimizer LR was not restored",
    )
    require(
        read(original / "experiment.json") == submission["dry_run"]["experiment"]
        and read(original / "wandb_run.json")["id"] == submission["training_wandb"]["id"],
        "Saved experiment or training W&B identity changed",
    )
    rows = [json.loads(line) for line in (original / "iterations.jsonl").read_text().splitlines()]
    require(
        [row["iteration"] for row in rows] == list(range(1, 752)),
        "Training updates are not contiguous 1..751",
    )
    require(
        all(
            row["slurm_job_id"] == submission["job_id"]
            and row["transitions"] == row["iteration"] * 12288
            for row in rows[100:]
        ),
        "Continuation iteration identity or transition budget mismatch",
    )
    manifest.update(
        status="evaluating",
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_hash,
        training_runtime=train_runtime,
        optimizer_state_restored=True,
        evaluations=[],
        evaluation_job_id=job,
    )
    save(manifest_path, manifest)
    for repeat in (1, 2):
        report_path = output / f"evaluation-repeat{repeat}.json"
        require(
            not report_path.exists(), "Existing evaluation output; inspect rather than duplicate"
        )
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
            report["checkpoint"] == str(checkpoint) and sha256(checkpoint) == checkpoint_hash,
            "Evaluation checkpoint changed",
        )
        require(
            report["episodes"] == 512
            and report["seed"] == 10000
            and report["split"] == "validation"
            and report["experiment"]["num_envs"] == 128
            and report["training_seed"] == 0
            and report["experiment"]["reward_profile"] == profile
            and report["experiment"]["success_state_sample"] == "current_qpos",
            "Corrected evaluation protocol mismatch",
        )
        require(
            len(report["per_variant"]) == 4
            and all(value["episodes"] == 128 for value in report["per_variant"].values()),
            "Unbalanced validation",
        )
        require(
            report["wandb_id"]
            == evaluation_submission["environment"][f"EVALUATION_WANDB_ID_{repeat}"],
            "Evaluation W&B identity mismatch",
        )
        audit = report["termination_audit"]
        require(
            audit["episodes"] == 512
            and all(
                audit[key] == 0
                for key in (
                    "success_hold_violations",
                    "success_distance_violations",
                    "success_three_sample_violations",
                    "success_qpos_above_distance_threshold",
                )
            ),
            "Physical success audit failed",
        )
        item = {
            "repeat": repeat,
            "path": str(report_path),
            "sha256": sha256(report_path),
            "wandb_id": report["wandb_id"],
            "wandb_url": report["wandb_url"],
            "successes": report["successes"],
            "episodes": 512,
        }
        manifest["evaluations"].append(item)
        if repeat == 1:
            require(
                bool(report["representative_capture"]),
                "Actual evaluated trajectories were not captured",
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
            records = read(media / "representatives.json")["records"]
            require(
                records
                and all(
                    Path(record[key]).is_file() and Path(record[key]).stat().st_size > 0
                    for record in records
                    for key in ("outside_video", "policy_video")
                ),
                "Captured videos missing",
            )
            item.update(
                media_manifest=str(media / "representatives.json"),
                media_manifest_sha256=sha256(media / "representatives.json"),
            )
        else:
            require(
                report["repeat_comparison"]["same_success_state_sample"],
                "Repeated success criterion differs",
            )
        save(manifest_path, manifest)
    manifest.update(status="complete", finished_at=now())
    runtime.update(status="complete", finished_at=now())
    save(manifest_path, manifest)
    save(runtime_path, runtime)
except BaseException as error:
    runtime.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
    manifest.update(status="failed", error=runtime["error"])
    save(runtime_path, runtime)
    save(manifest_path, manifest)
    raise
PY
