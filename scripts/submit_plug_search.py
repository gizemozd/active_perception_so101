"""Submit one bounded search case through train_plug.sbatch; never duplicate a manifest."""

import argparse
import json
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CASES = {
    "fixed_lr": {"LEARNING_RATE": "0.0001", "LR_SCHEDULE": "fixed", "ENTROPY_COEF": "0.003"},
    "entropy": {"LEARNING_RATE": "0.0003", "LR_SCHEDULE": "adaptive", "ENTROPY_COEF": "0.01"},
    "both": {"LEARNING_RATE": "0.0001", "LR_SCHEDULE": "fixed", "ENTROPY_COEF": "0.01"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    source, data = args.source_root.resolve(), args.data_root.resolve()
    outputs = data / "artifacts/hparam_search"
    run_id = secrets.token_hex(4)
    settings = {
        "PROJECT_ROOT": str(source),
        "PYTHONPATH": str(source / "src"),
        "NUM_ENVS": "512",
        "TOTAL_STEPS": "1228800",
        "FIXED_VIEW": "7",
        "MEMORY": "gru",
        "OCCLUSION": "clean",
        "JOB_TYPE": "search",
        "RUN_LABEL": args.case,
        "LOG_ROOT": str(data / "logs/hparam_search" / args.case),
        "RESULT_ROOT": str(outputs / args.case),
        "WANDB_ENTITY": "pgozdil-harvard-university",
        "WANDB_PROJECT": "active-perception-so101",
        "WANDB_RUN_GROUP": "plug-hparam-20261007",
        "WANDB_JOB_TYPE": "search",
        "WANDB_TAGS": f"restored-plug,search,wrist_static,view7,gru,seed0,{args.case}",
        "WANDB_MODE": "online",
        "WANDB_RUN_ID": run_id,
        **CASES[args.case],
    }
    env = dict(os.environ)
    for key in ("SLURM_STEPMGR", "WANDB_RESUME", "WANDB_USERNAME", "RESUME", "DRY_RUN"):
        env.pop(key, None)
    env.update(settings)
    command = [
        "sbatch",
        "--parsable",
        "--account=kempner_pgozdil_lab",
        "--partition=kempner_rtx",
        "--export=ALL",
        "--array=6",
        "--time=01:00:00",
        f"--job-name=plug-hp-{args.case}",
        f"--output={data}/logs/slurm/%x-%A_%a.out",
        f"--error={data}/logs/slurm/%x-%A_%a.err",
        "scripts/slurm/train_plug.sbatch",
    ]
    dry = subprocess.run(
        ["bash", "scripts/slurm/train_plug.sbatch"],
        cwd=source,
        env={**env, "DRY_RUN": "1", "SLURM_ARRAY_TASK_ID": "6"},
        capture_output=True,
        text=True,
        check=True,
    )
    config = json.loads(dry.stdout[dry.stdout.index("{") :])
    record = {
        "case": args.case,
        "source_root": str(source),
        "environment": settings,
        "command": command,
        "dry_run": config,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "wandb_url": f"https://wandb.ai/{settings['WANDB_ENTITY']}/{settings['WANDB_PROJECT']}/runs/{run_id}",
    }
    if args.submit:
        outputs.mkdir(parents=True, exist_ok=True)
        manifest = outputs / f"submission-{args.case}.json"
        record.update(
            submitted_at=datetime.now(timezone.utc).isoformat(), status="submission_reserved"
        )
        # Exclusive reservation survives interruption: inspect accounting before retrying.
        with manifest.open("x") as file:
            json.dump(record, file, indent=2)
        result = subprocess.run(command, cwd=source, env=env, capture_output=True, text=True)
        record.update(
            stdout=result.stdout.strip(), stderr=result.stderr.strip(), returncode=result.returncode
        )
        if result.returncode == 0:
            record.update(job_id=result.stdout.strip().split(";")[0] + "_6", status="submitted")
        else:
            record["status"] = "submission_failed_inspect_before_retry"
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        result.check_returncode()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
