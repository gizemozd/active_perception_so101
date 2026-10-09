#!/usr/bin/env bash
# Run only after the entry decision in reward_rescue_plan.json is approved.
# PROJECT_ROOT is the frozen 434c37a worktree; DATA_ROOT is the results checkout.
# DRY_RUN=1 validates the prepared CLI and prints the sequence without GPU work.
# Submission identity is reserved by the caller; runtime_manifest.json is exclusive.
#SBATCH --job-name=plug-reward-screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the frozen 434c37a worktree}"
: "${DATA_ROOT:?Set DATA_ROOT to the results checkout}"
export REWARD_REPAIR_SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
source "${PROJECT_ROOT}/scripts/slurm/common.sh"
export PYTHONPATH="${PROJECT_ROOT}/src"
"${PROJECT_ROOT}/.venv/bin/python" - "$PROJECT_ROOT" "$DATA_ROOT" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

source, data = (Path(value).resolve() for value in sys.argv[1:])
python = str(source / ".venv/bin/python")
dry = os.environ.get("DRY_RUN", "0") == "1"
job = os.environ.get("SLURM_JOB_ID")
expected_revision = "434c37af8a3e1fc82611d6bb13d3fb55f28a2987"
plan_path = data / "artifacts/plug_analysis/reward_rescue_plan.json"
plan = json.loads(plan_path.read_text())
result_root = data / "artifacts/plug_analysis/reward_repair"
log_root = data / "logs/reward_repair"
runtime_path = result_root / "runtime_manifest.json"
script_path = Path(os.environ["REWARD_REPAIR_SCRIPT_PATH"]).absolute()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def validate_source():
    require(git("rev-parse", "HEAD") == expected_revision, "Source must be frozen at 434c37a")
    require(not git("status", "--porcelain", "--untracked-files=no"), "Frozen source has tracked changes")
    for relative, expected in plan["source_files_sha256"].items():
        require(sha256(source / relative) == expected, f"Prepared source hash mismatch: {relative}")


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


validate_source()
require([case["case"] for case in plan["cases"]] == ["progress", "legacy_log_hold"], "Unexpected prepared pair")
require(plan["stage1_total_new_transitions"] == 2457600, "Unexpected pair budget")
require(plan["evaluation"] | {
    "split": "validation", "seed": 10000, "num_envs": 128, "episodes": 512,
    "balanced_per_variant": 128, "audit_termination": True, "repeats": 2,
    "success_state_sample": "current_qpos",
} == plan["evaluation"], "Unexpected prepared evaluation protocol")

actions = []
for case in plan["cases"]:
    profile = case["case"]
    output = result_root / profile
    original = case["command"]
    require(original[:3] == [".venv/bin/python", "-m", "active_perception_arms.train"], "Unexpected training executable")
    require("--resume" not in original and "--dry-run" not in original, "Screen must train fresh")
    require(original[original.index("--run-label") + 1] == profile + "_qpos", "Profile identity mismatch")
    require(original[original.index("--log-root") + 1] == "logs/reward_repair", "Unexpected planned log root")
    command = [python, *original[1:]]
    # Only storage paths change: execute from frozen source for truthful git metadata.
    command[command.index("--log-root") + 1] = str(log_root)
    command += ["--result", str(output / "training.json")]
    prepared = json.loads(subprocess.check_output(command + ["--dry-run"], cwd=source, text=True))
    for key in ("experiment", "optimization", "initial_std"):
        require(prepared[key] == case[key], f"Prepared CLI mismatch: {profile}/{key}")
    require(prepared["iterations"] == case["total_updates"] == 100, "Expected 100 updates")
    require(case["total_transitions"] == 1228800 and not prepared["training_started"], "Unexpected training budget")
    actions.append({"profile": profile, "planned_command": original, "training_command": command,
                    "output_directory": str(output), "updates": 100, "transitions": 1228800})

# These entry points must exist in the frozen source; --help performs no rollout.
for module in ("evaluate", "record_evaluation"):
    subprocess.run([python, "-m", f"active_perception_arms.{module}", "--help"],
                   cwd=source, check=True, stdout=subprocess.DEVNULL)

if dry:
    print(json.dumps({"source_revision": expected_revision, "source_files_sha256": plan["source_files_sha256"],
                      "actions": actions, "evaluation": plan["evaluation"],
                      "runtime_manifest": str(runtime_path), "sequential_gpus": 1,
                      "training_started": False}, indent=2))
    raise SystemExit(0)

require(job is not None, "Real execution requires a Slurm allocation")
require(source != data, "Real execution requires a distinct frozen source worktree")
require(os.environ.get("SLURM_NTASKS", "1") == "1", "Expected one Slurm task")
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
require(bool(visible) and len(visible.split(",")) == 1, "Expected exactly one visible GPU")
for action in actions:
    require(not Path(action["output_directory"]).exists(), f"Existing profile output: {action['profile']}")
result_root.mkdir(parents=True, exist_ok=True)
manifest = {
    "status": "running", "started_at": now(), "slurm_job_id": job,
    "source_root": str(source), "data_root": str(data), "source_revision": expected_revision,
    "source_files_sha256": plan["source_files_sha256"], "plan": plan,
    "plan_path": str(plan_path), "plan_sha256": sha256(plan_path),
    "script_path": str(script_path), "script_sha256": sha256(script_path),
    "submission_manifest": str(result_root / "submission.json"),
    "hostname": platform.node(), "python": sys.version, "sequential_gpus": 1,
    "allocation": {key: os.environ.get(key) for key in (
        "SLURM_JOB_ID", "SLURM_SUBMIT_DIR", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "CUDA_VISIBLE_DEVICES")},
    "planned_transitions": 2457600, "stages": [], "profiles": {},
    "interpretation": plan["interpretation"], "training_started": False,
}
# Atomic duplicate protection, including two invocations of the same submission.
with runtime_path.open("x") as stream:
    stream.write(json.dumps(manifest, indent=2) + "\n")


def environment(profile, stage):
    env = os.environ.copy()
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_NAME", "WANDB_USERNAME"):
        env.pop(key, None)
    env.update(WANDB_RUN_ID=uuid.uuid4().hex[:12], WANDB_RESUME="never",
               WANDB_RUN_GROUP=f"plug-reward-screen-{job}", WANDB_JOB_TYPE=stage,
               WANDB_TAGS=f"reward-repair,current-qpos,{profile},{stage}",
               EVALUATION_STAGE="reward-repair-100-updates", PYTHONPATH=str(source / "src"))
    return env


def run(command, env, stage, profile):
    validate_source()
    record = {"profile": profile, "stage": stage, "started_at": now(),
              "command": command, "cwd": str(source), "wandb_id": env["WANDB_RUN_ID"]}
    manifest["stages"].append(record)
    save(runtime_path, manifest)
    print(json.dumps(record), flush=True)
    result = subprocess.run(["srun", "--ntasks=1", "--chdir", str(source), *command],
                            cwd=source, env=env)
    record.update(finished_at=now(), returncode=result.returncode)
    save(runtime_path, manifest)
    require(result.returncode == 0, f"{profile}/{stage} failed with exit code {result.returncode}")
    validate_source()


try:
    log_root.mkdir(parents=True, exist_ok=True)
    for action, case in zip(actions, plan["cases"], strict=True):
        profile = case["case"]
        output = Path(action["output_directory"])
        output.mkdir(exist_ok=False)
        profile_manifest = {"profile": profile, "slurm_job_id": job,
                            "source_revision": expected_revision,
                            "source_files_sha256": plan["source_files_sha256"],
                            "planned_command": case["command"], "training_command": action["training_command"],
                            "training_updates": 100, "training_transitions": 1228800,
                            "evaluations": [], "status": "training"}
        manifest["profiles"][profile] = profile_manifest
        save(output / "run_manifest.json", profile_manifest)
        before = {path.name for path in log_root.iterdir() if path.is_dir()}
        train_env = environment(profile, "search")
        manifest["training_started"] = True
        run(action["training_command"], train_env, "training", profile)
        prefix = f"plug_wrist_static_clean_gru_v7_s0_n512_{profile}_qpos_"
        created = [path for path in log_root.iterdir()
                   if path.is_dir() and path.name not in before and path.name.startswith(prefix)]
        require(len(created) == 1, f"Expected exactly one newly created {profile} run, found {len(created)}")
        run_directory = created[0]
        runtime = read(run_directory / "runtime.json")
        require(runtime["slurm_job_id"] == job and runtime["git_revision"] == expected_revision
                and runtime["resume"] is None, "Training run identity or source mismatch")
        require(read(run_directory / "experiment.json") == case["experiment"], "Saved experiment mismatch")
        wandb = read(run_directory / "wandb_run.json")
        require(wandb["id"] == train_env["WANDB_RUN_ID"], "Training W&B identity mismatch")
        rows = [json.loads(line) for line in (run_directory / "iterations.jsonl").read_text().splitlines()]
        require([row["iteration"] for row in rows] == list(range(1, 101)), "Incomplete 100-update training")
        require(all(row["slurm_job_id"] == job and row["transitions"] == row["iteration"] * 12288
                    for row in rows), "Training segment identity or transition budget mismatch")
        training = read(output / "training.json")
        require(training["actual_transitions"] == training["final_cumulative_transitions"] == 1228800,
                "Incomplete training transition budget")
        checkpoint = run_directory / "model_99.pt"
        require(checkpoint.is_file(), "Expected model_99.pt from the identified new run")
        import torch
        require(torch.load(checkpoint, map_location="cpu", weights_only=False)["iter"] == 99,
                "Checkpoint has unexpected iteration index")
        checkpoint_hash = sha256(checkpoint)
        profile_manifest.update(status="evaluating", run_directory=str(run_directory),
                                checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                                training_wandb=wandb, training_runtime=runtime)
        save(output / "run_manifest.json", profile_manifest)
        for repeat in (1, 2):
            report_path = output / f"evaluation-repeat{repeat}.json"
            evaluation = [python, "-m", "active_perception_arms.evaluate", str(checkpoint),
                          "--episodes", "512", "--num-envs", "128", "--seed", "10000",
                          "--split", "validation", "--success-state-sample", "current_qpos",
                          "--audit-termination", "--output", str(report_path), "--wandb"]
            if repeat == 1:
                evaluation += ["--capture-representatives", str(output / "evaluation-repeat1-traces")]
            else:
                evaluation += ["--reference-report", str(output / "evaluation-repeat1.json")]
            eval_env = environment(profile, f"validation-repeat{repeat}")
            run(evaluation, eval_env, f"validation-repeat{repeat}", profile)
            report = read(report_path)
            require(report["checkpoint"] == str(checkpoint) and sha256(checkpoint) == checkpoint_hash,
                    "Evaluation checkpoint identity changed")
            require(report["episodes"] == 512 and report["seed"] == 10000
                    and report["split"] == "validation" and report["training_seed"] == 0,
                    "Evaluation protocol mismatch")
            require(report["experiment"]["success_state_sample"] == "current_qpos"
                    and report["experiment"]["reward_profile"] == profile
                    and report["experiment"]["num_envs"] == 128, "Evaluation configuration mismatch")
            require(all(row["episodes"] == 128 for row in report["per_variant"].values())
                    and len(report["per_variant"]) == 4, "Validation variant balance mismatch")
            require(report["wandb_id"] == eval_env["WANDB_RUN_ID"], "Evaluation W&B identity mismatch")
            audit = report["termination_audit"]
            require(audit["episodes"] == 512 and all(audit[key] == 0 for key in (
                "success_hold_violations", "success_distance_violations",
                "success_three_sample_violations", "success_qpos_above_distance_threshold")),
                "Corrected physical success audit failed")
            if repeat == 2:
                require(report["repeat_comparison"]["same_success_state_sample"], "Repeat criterion mismatch")
            item = {"repeat": repeat, "path": str(report_path), "sha256": sha256(report_path),
                    "wandb_id": report["wandb_id"], "wandb_url": report["wandb_url"],
                    "successes": report["successes"], "episodes": report["episodes"]}
            profile_manifest["evaluations"].append(item)
            save(output / "run_manifest.json", profile_manifest)
            if repeat == 1:
                require(bool(report["representative_capture"]), "Actual trajectory capture missing")
                media = output / "evaluation-repeat1-videos"
                run([python, "-m", "active_perception_arms.record_evaluation", str(report_path),
                     "--output", str(media), "--wandb"], eval_env, "capture-videos", profile)
                media_manifest = read(media / "representatives.json")
                require(media_manifest["records"], "Representative videos missing")
                for record in media_manifest["records"]:
                    require(all(Path(record[key]).is_file() and Path(record[key]).stat().st_size > 0
                                for key in ("outside_video", "policy_video")), "Captured video missing")
                item["media_manifest"] = str(media / "representatives.json")
                item["media_manifest_sha256"] = sha256(media / "representatives.json")
                save(output / "run_manifest.json", profile_manifest)
        profile_manifest.update(status="complete", finished_at=now())
        save(output / "run_manifest.json", profile_manifest)
        save(runtime_path, manifest)
    manifest.update(status="complete", finished_at=now())
    save(runtime_path, manifest)
except BaseException as error:
    manifest.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
    save(runtime_path, manifest)
    raise
PY
