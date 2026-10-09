"""Prepare a matched reward continuation; submit only after the screen/HP entry decision."""

import argparse
import hashlib
import json
import os
import secrets
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from submit_tuning_continuation import submit_reserved

BOOTSTRAP_FIX = "11a591f7a2129484d555caf115fcdbf1a709a0c9"
SOURCE_CHANGE = "critic_bootstrap_memory_restore_v1"
PROFILES = ("progress", "legacy_log_hold")
TOTAL_UPDATES = 751
TOTAL_TRANSITIONS = 9228288
ADDITIONAL_TRANSITIONS = 7999488
SOURCE_FILES = (
    "src/active_perception_arms/config.py",
    "src/active_perception_arms/policy.py",
    "src/active_perception_arms/train.py",
    "src/active_perception_arms/mdp.py",
    "src/active_perception_arms/environment.py",
    "src/active_perception_arms/telemetry.py",
    "src/active_perception_arms/evaluate.py",
    "src/active_perception_arms/capture_evaluation.py",
    "src/active_perception_arms/record_evaluation.py",
    "scripts/slurm/common.sh",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def validate_source(source, data):
    require(source != data, "Use a distinct frozen source worktree")
    require(
        not git(source, "status", "--porcelain", "--untracked-files=no"),
        "Frozen source has tracked changes",
    )
    subprocess.run(
        ["git", "-C", str(source), "merge-base", "--is-ancestor", BOOTSTRAP_FIX, "HEAD"], check=True
    )
    committed_policy = subprocess.check_output(
        ["git", "-C", str(data), "show", "HEAD:src/active_perception_arms/policy.py"]
    )
    require(
        sha256(source / SOURCE_FILES[1]) == hashlib.sha256(committed_policy).hexdigest(),
        "Frozen source must contain the newest committed critic-bootstrap implementation",
    )
    require((source / ".venv/bin/python").is_file(), "Frozen source needs its environment")
    return git(source, "rev-parse", "HEAD")


def clean_environment():
    env = dict(os.environ)
    # Submission options belong to this helper; inherited array defaults could
    # otherwise turn one profile into several simultaneous allocations.
    for key in tuple(env):
        if key.startswith("SBATCH_"):
            env.pop(key)
    for key in (
        "DRY_RUN",
        "SLURM_STEPMGR",
        "WANDB_RUN_ID",
        "WANDB_RESUME",
        "WANDB_RUN_NAME",
        "WANDB_USERNAME",
        "RESUME",
        "REWARD_PROFILE",
        "INITIAL_STD",
        "SUCCESS_STATE_SAMPLE",
        "LEARNING_RATE",
        "LR_SCHEDULE",
        "ENTROPY_COEF",
        "REFERENCE_REPORT",
    ):
        env.pop(key, None)
    return env


# Embedded in sbatch --wrap: validate again when the allocation starts, before
# train.py modifies saved runtime/runner metadata. The original checkpoint stays.
PREFLIGHT = r"""
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
source = Path(record['source_root'])
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == record['source_revision']
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
for path, expected in record['source_files_sha256'].items():
    assert sha(source / path) == expected, path
for path, expected in record['resume_files_sha256'].items():
    assert sha(Path(record['run_directory']) / path) == expected, path
assert not Path(record['expected_final_checkpoint']).exists(), 'Existing model_750; inspect rather than duplicate'
wandb = json.loads((Path(record['run_directory']) / 'wandb_run.json').read_text())
assert wandb['id'] == record['training_wandb']['id']
runtime = dict(slurm_job_id=os.environ['SLURM_JOB_ID'], source_revision=record['source_revision'],
               source_change=record['source_change'], started_at=datetime.now(timezone.utc).isoformat(),
               resume_checkpoint=record['resume_checkpoint'], resume_checkpoint_sha256=record['resume_checkpoint_sha256'],
               allocation={k:os.environ.get(k) for k in ('SLURM_JOB_ACCOUNT','SLURM_JOB_PARTITION','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE','CUDA_VISIBLE_DEVICES')})
with (Path(record['output_directory']) / 'training-runtime.json').open('x') as f:
    json.dump(runtime, f, indent=2)
"""


def prepare(profile, source, data, revision):
    import torch

    root = data / "artifacts/plug_analysis/reward_repair"
    screen = read(root / profile / "run_manifest.json")
    checkpoint = Path(screen["checkpoint"])
    require(
        checkpoint.name == "model_99.pt" and checkpoint.parent == Path(screen["run_directory"]),
        "Screen checkpoint identity mismatch",
    )
    require(sha256(checkpoint) == screen["checkpoint_sha256"], "Screen checkpoint hash changed")
    saved_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved_checkpoint["iter"] == 99, "Expected the completed 100-update checkpoint")
    optimizer = saved_checkpoint["optimizer_state_dict"]
    rates = {group["lr"] for group in optimizer["param_groups"]}
    require(
        len(rates) == 1 and optimizer["state"], "Expected populated Adam state with one shared LR"
    )
    original = checkpoint.parent
    experiment = read(original / "experiment.json")
    runner = read(original / "runner.json")
    training = read(root / profile / "training.json")
    wandb = read(original / "wandb_run.json")
    require(
        experiment == training["experiment"]
        and experiment["reward_profile"] == profile
        and experiment["success_state_sample"] == "current_qpos",
        "Screen experiment mismatch",
    )
    require(
        experiment["num_envs"] == 512 and training["final_cumulative_transitions"] == 1228800,
        "Expected the N512 100-update screen",
    )
    require(wandb["id"] == screen["training_wandb"]["id"], "Saved training W&B identity changed")
    require(runner["max_iterations"] == 100, "Original runner has already been extended")
    output = root / "continuation" / profile
    require(
        not (original / "model_750.pt").exists(),
        "Final checkpoint already exists; inspect rather than duplicate",
    )
    # Inherit profile, physical criterion, std and PPO knobs from saved files.
    dropped = {
        "--iterations",
        "--log-root",
        "--initial-std",
        "--learning-rate",
        "--lr-schedule",
        "--entropy-coef",
        "--reward-profile",
        "--success-state-sample",
        "--run-label",
    }
    arguments = []
    command = screen["planned_command"]
    require(
        command[:3] == [".venv/bin/python", "-m", "active_perception_arms.train"],
        "Unexpected screen command",
    )
    iterator = iter(command[3:])
    for token in iterator:
        if token in dropped:
            next(iterator)
        else:
            arguments.append(token)
    train_command = [
        str(source / ".venv/bin/python"),
        "-m",
        "active_perception_arms.train",
        *arguments,
        "--iterations",
        "751",
        "--resume",
        str(checkpoint),
        "--result",
        str(output / "training.json"),
    ]
    env = clean_environment()
    env.update(
        PROJECT_ROOT=str(source),
        PYTHONPATH=str(source / "src"),
        WANDB_ENTITY=wandb["entity"],
        WANDB_PROJECT=wandb["project"],
        WANDB_RUN_ID=wandb["id"],
        WANDB_RESUME="must",
        WANDB_MODE="online",
        WANDB_RUN_GROUP=f"plug-reward-screen-{screen['slurm_job_id']}",
        WANDB_JOB_TYPE="search",
        WANDB_TAGS=f"reward-repair,continuation,{SOURCE_CHANGE},{profile}",
    )
    dry = json.loads(
        subprocess.check_output(train_command + ["--dry-run"], cwd=source, env=env, text=True)
    )
    require(
        dry["experiment"] == experiment and dry["iterations"] == TOTAL_UPDATES,
        "Resume experiment/budget mismatch",
    )
    require(
        dry["initial_std"] == runner["actor"]["distribution_cfg"]["init_std"],
        "Resume std inheritance mismatch",
    )
    require(
        dry["optimization"]
        == {key: runner["algorithm"][key] for key in ("learning_rate", "schedule", "entropy_coef")},
        "Resume optimization inheritance mismatch",
    )
    # Check the whole model/algorithm construction, not just the three CLI knobs.
    architecture_check = """
import json, sys
from dataclasses import asdict
from active_perception_arms.train import parser, experiment_from_args, initial_std_from_args, optimization_from_args
from active_perception_arms.policy import runner_cfg
args = parser().parse_args(sys.argv[2:])
current = asdict(runner_cfg(experiment_from_args(args), args.iterations))
current['actor']['distribution_cfg']['init_std'] = initial_std_from_args(args)
current['algorithm'].update(optimization_from_args(args))
current = json.loads(json.dumps(current))
saved = json.load(open(sys.argv[1]))
for key in ('actor', 'critic', 'algorithm', 'obs_groups', 'num_steps_per_env', 'seed'):
    assert current[key] == saved[key], key
"""
    subprocess.run(
        [
            train_command[0],
            "-c",
            architecture_check,
            str(original / "runner.json"),
            *train_command[3:],
        ],
        cwd=source,
        env=env,
        check=True,
    )
    source_hashes = {path: sha256(source / path) for path in SOURCE_FILES}
    source_history = [
        {"revision": screen["source_revision"], "updates": 100, "transitions": 1228800},
        {"revision": revision, "updates": 651, "transitions": ADDITIONAL_TRANSITIONS},
    ]
    record = {
        "profile": profile,
        "source_root": str(source),
        "data_root": str(data),
        "source_revision": revision,
        "current_source_revision": revision,
        "initial_source_revision": screen["source_revision"],
        "source_history": source_history,
        "source_change": SOURCE_CHANGE,
        "source_files_sha256": source_hashes,
        "source_change_note": "Both profiles resume after 100 original-source updates with recurrent critic bootstrap memory restored. This is a mixed-source continuation, not a fresh corrected 751-update run.",
        "run_directory": str(original),
        "resume_checkpoint": str(checkpoint),
        "resume_checkpoint_sha256": sha256(checkpoint),
        "resume_files_sha256": {
            name: sha256(original / name)
            for name in ("model_99.pt", "experiment.json", "runner.json", "wandb_run.json")
        },
        "resume_optimizer_lr": rates.pop(),
        "optimizer_state_restore_required": True,
        "training_wandb": wandb,
        "training_command": train_command,
        "dry_run": dry,
        "training_updates": TOTAL_UPDATES,
        "training_transitions": TOTAL_TRANSITIONS,
        "additional_updates": 651,
        "additional_transitions": ADDITIONAL_TRANSITIONS,
        "expected_final_checkpoint": str(original / "model_750.pt"),
        "output_directory": str(output),
        "original_screen_manifest": str(root / profile / "run_manifest.json"),
        "original_screen_manifest_sha256": sha256(root / profile / "run_manifest.json"),
        "resume_state_note": "Weights, Adam and adaptive LR are restored. Simulator, RNG and GRU episode state reset at resume; actor memory is not checkpointed.",
        "evaluation": {
            "episodes": 512,
            "num_envs": 128,
            "seed": 10000,
            "split": "validation",
            "repeats": 2,
            "success_state_sample": "current_qpos",
            "audit_termination": True,
        },
    }
    settings = {
        key: env[key]
        for key in (
            "PROJECT_ROOT",
            "PYTHONPATH",
            "WANDB_ENTITY",
            "WANDB_PROJECT",
            "WANDB_RUN_ID",
            "WANDB_RESUME",
            "WANDB_MODE",
            "WANDB_RUN_GROUP",
            "WANDB_JOB_TYPE",
            "WANDB_TAGS",
        )
    }
    return record, env, settings


def queue_evaluation(record, source, data, submit):
    output = Path(record["output_directory"])
    script = data / "scripts/slurm/reward_continuation_eval.sbatch"
    settings = {
        "PROJECT_ROOT": str(source),
        "DATA_ROOT": str(data),
        "PYTHONPATH": str(source / "src"),
        "CONTINUATION_MANIFEST": str(output / "submission.json"),
        "WANDB_MODE": "online",
        "WANDB_RUN_GROUP": "plug-reward-bootstrap-continuation",
        "WANDB_ENTITY": record["training_wandb"]["entity"],
        "WANDB_PROJECT": record["training_wandb"]["project"],
        "EVALUATION_STAGE": SOURCE_CHANGE,
        "EVALUATION_WANDB_ID_1": secrets.token_hex(6),
        "EVALUATION_WANDB_ID_2": secrets.token_hex(6),
    }
    command = [
        "sbatch",
        "--parsable",
        "--account=kempner_pgozdil_lab",
        "--partition=kempner_rtx",
        "--export=ALL",
        "--exclude=holygpu7c2313",
        f"--dependency=afterok:{record.get('job_id', '<training-job>')}",
        f"--job-name=plug-reward751-eval-{record['profile']}",
        f"--output={data}/logs/slurm/%x-%j.out",
        f"--error={data}/logs/slurm/%x-%j.err",
        str(script),
    ]
    evaluation = {
        "profile": record["profile"],
        "training_job_id": record.get("job_id"),
        "source_root": str(source),
        "source_revision": record["source_revision"],
        "source_files_sha256": record["source_files_sha256"],
        "source_history": record["source_history"],
        "source_change": SOURCE_CHANGE,
        "script_sha256": sha256(script),
        "environment": settings,
        "command": command,
    }
    if submit:
        env = clean_environment()
        env.update(settings)
        evaluation = submit_reserved(
            output / "evaluation-submission.json", evaluation, command, source, env
        )
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--submit", action="store_true", help="Use only after completed screen and HP review"
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Queue a missing evaluation for a verified existing training submission",
    )
    args = parser.parse_args()
    data, source = args.data_root.resolve(), args.source_root.resolve()
    revision = validate_source(source, data)
    output = data / "artifacts/plug_analysis/reward_repair/continuation" / args.profile
    manifest_path = output / "submission.json"
    if args.evaluation_only:
        record = read(manifest_path)
        require(
            record["status"] == "submitted" and str(record["job_id"]).isdecimal(),
            "Inspect uncertain training reservation before queueing evaluation",
        )
        require(
            record["source_root"] == str(source) and record["source_revision"] == revision,
            "Existing submission source mismatch",
        )
        require(
            not (output / "evaluation-submission.json").exists(),
            "Evaluation reservation exists; inspect rather than resubmit",
        )
        evaluation = queue_evaluation(record, source, data, args.submit)
        print(
            json.dumps(
                {"training": record, "evaluation": evaluation, "training_started": False}, indent=2
            )
        )
        return
    require(
        not manifest_path.exists(), "Existing training reservation; inspect rather than resubmit"
    )
    require(not output.exists(), "Existing continuation output; inspect rather than duplicate")
    record, env, settings = prepare(args.profile, source, data, revision)
    preflight = [str(source / ".venv/bin/python"), "-c", PREFLIGHT, str(manifest_path)]
    wrap = (
        "set -euo pipefail\nsource "
        + shlex.quote(str(source / "scripts/slurm/common.sh"))
        + "\n"
        + shlex.join(preflight)
        + "\nexec "
        + shlex.join(["srun", "--ntasks=1", "--chdir", str(source), *record["training_command"]])
    )
    command = [
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
        f"--job-name=plug-reward751-{args.profile}",
        f"--output={data}/logs/slurm/%x-%j.out",
        f"--error={data}/logs/slurm/%x-%j.err",
        "--wrap",
        "exec bash -c " + shlex.quote(wrap),
    ]
    record.update(
        environment=settings,
        command=command,
        prepared_at=datetime.now(timezone.utc).isoformat(),
        helper_sha256=sha256(Path(__file__)),
        max_parallel_extension_gpus=2,
    )
    evaluation = queue_evaluation(record, source, data, False)
    if not args.submit:
        print(
            json.dumps(
                {"training": record, "evaluation": evaluation, "training_started": False}, indent=2
            )
        )
        return
    screen_root = data / "artifacts/plug_analysis/reward_repair"
    require(
        read(screen_root / "runtime_manifest.json")["status"] == "complete",
        "Wait for the complete matched 100-update screen",
    )
    require(
        all(
            read(screen_root / profile / "run_manifest.json")["status"] == "complete"
            for profile in PROFILES
        ),
        "Both screen profiles must finish before extension",
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_manifest.json").write_text(
        json.dumps({**record, "status": "prepared"}, indent=2) + "\n"
    )
    record = submit_reserved(manifest_path, record, command, source, env)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {**record, "status": "training_submitted", "training_job_id": record["job_id"]},
            indent=2,
        )
        + "\n"
    )
    evaluation = queue_evaluation(record, source, data, True)
    record["evaluation_job_id"] = evaluation["job_id"]
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {**record, "status": "training_submitted", "training_job_id": record["job_id"]},
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "profile": args.profile,
                "training_job": record["job_id"],
                "evaluation_job": evaluation["job_id"],
                "additional_updates": 651,
                "source_change": SOURCE_CHANGE,
                "wandb_url": record["training_wandb"]["url"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
