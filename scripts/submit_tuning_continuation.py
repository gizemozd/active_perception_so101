"""Resume a prepared tuning case without duplicating its original budget or identity."""

import argparse
import json
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def revision(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def submit_reserved(path, record, command, cwd, env):
    """An uncertain submission keeps its reservation and requires accounting inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record.update(status="submission_reserved", reserved_at=datetime.now(timezone.utc).isoformat())
    with path.open("x") as file:
        json.dump(record, file, indent=2)
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    record.update(
        stdout=result.stdout.strip(), stderr=result.stderr.strip(), returncode=result.returncode
    )
    if result.returncode == 0:
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            record["status"] = "uncertain_response_inspect_accounting"
            path.write_text(json.dumps(record, indent=2) + "\n")
            raise RuntimeError("Unexpected sbatch response; reservation retained")
        record.update(status="submitted", job_id=job_id)
    else:
        record["status"] = "submission_failed_inspect_before_retry"
    path.write_text(json.dumps(record, indent=2) + "\n")
    result.check_returncode()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("fixed_lr", "entropy", "both"))
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args()
    if args.evaluation_only and not args.submit:
        parser.error("--evaluation-only requires --submit")
    data = args.data_root.resolve()
    eval_root = args.evaluation_root.resolve()
    plan = json.loads((data / "artifacts/hparam_search/next_stage_plan.json").read_text())
    case = next(row for row in plan["cases"] if row["case"] == args.case)
    source = Path(case["source_root"])
    if revision(source) != case["source_revision"]:
        raise RuntimeError("Frozen training revision differs from prepared plan")
    for root in (source, eval_root):
        if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=root, text=True):
            raise RuntimeError(f"Tracked source changes in frozen checkout {root}")
    output_root = data / "artifacts/hparam_search/continuation"
    manifest = output_root / f"submission-{args.case}.json"
    if manifest.exists() and not args.evaluation_only:
        raise FileExistsError(f"Existing reservation: {manifest}; inspect rather than resubmit")
    settings = case["environment"]
    env = dict(os.environ)
    for key in ("SLURM_STEPMGR", "DRY_RUN", "REFERENCE_REPORT", "WANDB_USERNAME"):
        env.pop(key, None)
    env.update(settings)
    command = case["command"]
    if not args.evaluation_only:
        import torch

        checkpoint = torch.load(settings["RESUME"], map_location="cpu", weights_only=True)
        if checkpoint["iter"] != 99:
            raise RuntimeError("Expected the completed 100-update checkpoint")
        dry = subprocess.run(
            ["bash", "scripts/slurm/train_plug.sbatch"],
            cwd=source,
            env={**env, "DRY_RUN": "1", "SLURM_ARRAY_TASK_ID": "6"},
            capture_output=True,
            text=True,
            check=True,
        )
        config = json.loads(dry.stdout[dry.stdout.index("{") :])
        if config["iterations"] != 751 or config["experiment"] != case["dry_run"]["experiment"]:
            raise RuntimeError("Resume dry-run differs from prepared experiment/budget")
        record = {
            **case,
            "dry_run": config,
            "additional_transitions": plan["additional_per_case"],
            "rationale": "Final original runs have low validation success and all fail xm; controlled optimization repair before a larger condition study.",
            "evaluation_source_root": str(eval_root),
            "evaluation_source_revision": revision(eval_root),
        }
        if not args.submit:
            print(
                json.dumps(
                    {"case": args.case, "dry_run": config, "training_started": False}, indent=2
                )
            )
            return
        record = submit_reserved(manifest, record, command, source, env)
    else:
        record = json.loads(manifest.read_text())
        if record["status"] != "submitted":
            raise RuntimeError("Training submission is not verified")
    evaluation_settings = {
        "PROJECT_ROOT": str(eval_root),
        "PYTHONPATH": str(eval_root / "src"),
        "CHECKPOINT": case["expected_final_checkpoint"],
        "OUTPUT": case["evaluation_output"],
        "NUM_ENVS": "128",
        "EPISODES": "512",
        "SPLIT": "validation",
        "RECORD_CAPTURE": "1",
        "RECORD_VIDEO": "1",
        "WANDB_MODE": "online",
        "WANDB_RUN_GROUP": "plug-hparam-20261007",
        "EVALUATION_STAGE": "search",
        "WANDB_ENTITY": settings["WANDB_ENTITY"],
        "WANDB_PROJECT": settings["WANDB_PROJECT"],
        "WANDB_RUN_ID": secrets.token_hex(4),
    }
    evaluation_env = dict(os.environ)
    for key in ("SLURM_STEPMGR", "DRY_RUN", "WANDB_RESUME", "REFERENCE_REPORT", "RESUME"):
        evaluation_env.pop(key, None)
    evaluation_env.update(evaluation_settings)
    evaluation_command = [
        "sbatch",
        "--parsable",
        "--account=kempner_pgozdil_lab",
        "--partition=kempner_rtx",
        "--export=ALL",
        f"--dependency=afterok:{record['job_id']}",
        "--exclude=holygpu7c2313",
        f"--job-name=plug-hp2-eval-{args.case}",
        f"--output={data}/logs/slurm/%x-%j.out",
        "scripts/slurm/evaluate.sbatch",
    ]
    evaluation_record = submit_reserved(
        output_root / f"evaluation-submission-{args.case}.json",
        {
            "case": args.case,
            "training_job_id": record["job_id"],
            "environment": evaluation_settings,
            "command": evaluation_command,
            "source_root": str(eval_root),
            "source_revision": revision(eval_root),
        },
        evaluation_command,
        eval_root,
        evaluation_env,
    )
    record["evaluation_job_id"] = evaluation_record["job_id"]
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "case": args.case,
                "training_job": record["job_id"],
                "evaluation_job": evaluation_record["job_id"],
                "remaining_updates": 651,
                "wandb_url": case["wandb_url"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
