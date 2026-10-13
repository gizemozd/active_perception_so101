"""Execute one frozen, bounded plug reward pilot and its checkpoint evaluations."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out = Path(plan["output"])
    source = Path(plan["source"])
    for relative, expected in plan["source_sha256"].items():
        with (source / relative).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, relative
    status = dict(started_at=now(), job_id=os.environ.get("SLURM_JOB_ID"))

    def save():
        temp = out / "state.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(out / "state.json")

    def run(command, stage):
        status.update(status=stage, updated_at=now(), command=command)
        save()
        with (out / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(time=now(), command=command)) + "\n")
        subprocess.run(command, cwd=source, env=env, check=True)

    env = os.environ | {
        "PYTHONPATH": str(source / "src"),
        "MUJOCO_GL": "egl",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "WANDB_RUN_GROUP": "plug-insertion-reward-pilot",
        "WANDB_TAGS": "plug,plug_insertion,wrist_static,short-pilot",
        "XDG_CACHE_HOME": "/tmp/plug-reward-" + str(os.environ.get("SLURM_JOB_ID", "local")),
    }
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_NAME"):
        env.pop(key, None)
    try:
        run(plan["command"], "training")
        training = json.loads((out / "training.json").read_text())
        assert training["final_cumulative_transitions"] == plan["transitions"]
        directories = list(Path(plan["log_root"]).glob("*/experiment.json"))
        assert len(directories) == 1
        run_dir = directories[0].parent
        status["run_directory"] = str(run_dir)
        for index in (0, plan["iterations"] - 1):
            checkpoint = run_dir / f"model_{index}.pt"
            command = [
                plan["python"],
                "-m",
                "active_perception_arms.evaluate",
                str(checkpoint),
                "--episodes",
                "128",
                "--num-envs",
                "128",
                "--seed",
                "30000",
                "--split",
                "validation",
                "--occlusion",
                "clean",
                "--audit-termination",
                "--output",
                str(out / f"evaluation-{index + 1}.json"),
            ]
            run(command, f"evaluating_{index + 1}")
        status.update(status="completed", completed_at=now(), transitions=plan["transitions"])
        save()
    except BaseException as error:
        status.update(status="failed", error=repr(error), updated_at=now())
        save()
        raise


if __name__ == "__main__":
    main()
