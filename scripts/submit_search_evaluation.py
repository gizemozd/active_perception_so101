"""Queue one 512-episode validation after a submitted search case finishes."""

import argparse
import json
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("fixed_lr", "entropy", "both"))
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    outputs = root / "artifacts/hparam_search"
    parent = json.loads((outputs / f"submission-{args.case}.json").read_text())
    parent_id = parent["job_id"].split("_")[0]
    matches = []
    for path in (root / "logs/hparam_search" / args.case).glob("*/runtime.json"):
        if json.loads(path.read_text())["slurm_job_id"] == parent_id:
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one started training directory for {parent_id}, found {matches}"
        )
    run_dir = matches[0]
    settings = {
        "PROJECT_ROOT": str(root),
        "PYTHONPATH": str(root / "src"),
        "CHECKPOINT": str(run_dir / "model_99.pt"),
        "NUM_ENVS": "128",
        "EPISODES": "512",
        "SPLIT": "validation",
        "RECORD_CAPTURE": "1",
        "RECORD_VIDEO": "1",
        "OUTPUT": str(outputs / args.case / "evaluation.json"),
        "WANDB_RUN_GROUP": "plug-hparam-20261007",
        "EVALUATION_STAGE": "search",
        "WANDB_MODE": "online",
        "WANDB_RUN_ID": secrets.token_hex(4),
    }
    env = dict(os.environ)
    for key in ("SLURM_STEPMGR", "WANDB_RESUME", "REFERENCE_REPORT", "DRY_RUN"):
        env.pop(key, None)
    env.update(settings)
    command = [
        "sbatch",
        "--parsable",
        "--account=kempner_pgozdil_lab",
        "--partition=kempner_rtx",
        "--export=ALL",
        f"--dependency=afterok:{parent['job_id']}",
        f"--job-name=plug-hp-eval-{args.case}",
        "scripts/slurm/evaluate.sbatch",
    ]
    record = {
        "case": args.case,
        "training_job_id": parent["job_id"],
        "command": command,
        "environment": settings,
        "working_directory": str(root),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "submission_reserved",
    }
    manifest = outputs / f"evaluation-submission-{args.case}.json"
    with manifest.open("x") as file:
        json.dump(record, file, indent=2)
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    record.update(
        stdout=result.stdout.strip(), stderr=result.stderr.strip(), returncode=result.returncode
    )
    if result.returncode == 0:
        record.update(job_id=result.stdout.strip().split(";")[0], status="submitted")
    else:
        record["status"] = "submission_failed_inspect_before_retry"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    result.check_returncode()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
