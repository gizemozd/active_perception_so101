#!/usr/bin/env python3
"""Prepare a hash-pinned source snapshot; run GPU validation or a bounded recipe pair."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def prepare(root, manifest_path):
    if manifest_path.exists():
        raise FileExistsError("Prepared manifest exists; use a new path for a new experiment")
    output = manifest_path.parent.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (
        root / ".cache" / f"plug-recipe-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    shutil.copytree(
        root / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in ("validate_plug_repair_gpu.py", "run_plug_recipe_repair.py"):
        shutil.copy2(root / "scripts" / name, source / name)
    (source / "working-tree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=root)
    )
    files = {
        str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*")) if p.is_file()
    }
    python = str(root / ".venv/bin/python")
    cases = []
    for yaw in (False, True):
        name = f"legacy_ff_yaw{int(yaw)}"
        command = [
            python,
            "-m",
            "active_perception_arms.train",
            "--control-revision",
            "legacy_v1",
            "--task",
            "plug",
            "--condition",
            "wrist_static",
            "--fixed-view",
            "7",
            "--seed",
            "0",
            "--policy-profile",
            "legacy_ff",
            "--reward-profile",
            "legacy_log_hold",
            "--num-envs",
            "512",
            "--iterations",
            "751",
            "--run-label",
            name,
            "--log-root",
            str(root / "logs" / "recipe_repair"),
            "--job-type",
            "search",
            "--result",
            str(output / name / "training.json"),
        ]
        if yaw:
            command.append("--plug-wrist-yaw")
        prepared = json.loads(
            subprocess.check_output(
                command + ["--dry-run"],
                cwd=root,
                text=True,
                env=os.environ | {"PYTHONPATH": str(source / "src")},
            )
        )
        cases.append(
            {"name": name, "command": command, "prepared": prepared, "transitions": 9228288}
        )
    manifest = {
        "prepared_at": now(),
        "root": str(root),
        "source": str(source),
        "python": python,
        "output": str(output),
        "source_files_sha256": files,
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_note": "Uncommitted working-tree snapshot, identified by full file hashes and patch; base revision alone is insufficient.",
        "cases": cases,
        "total_training_transitions": 18456576,
        "max_concurrent_gpus": 1,
        "evaluation": {
            "episodes": 512,
            "num_envs": 128,
            "seed": 10000,
            "repeats": 2,
            "split": "validation",
            "min_successes_per_variant_per_repeat": 103,
        },
        "interpretation": "Legacy-style FF/log recipe, not exact old reproduction. Yaw case also shapes and scores orientation. Single training seed, no automatic extension or camera-condition sweep.",
    }
    with manifest_path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "source": str(source),
                "cases": [c["name"] for c in cases],
            },
            indent=2,
        )
    )


def execute(manifest_path, stage):
    plan = json.loads(manifest_path.read_text())
    source, output = Path(plan["source"]), Path(plan["output"])
    assert os.environ.get("SLURM_JOB_ID"), "Real execution requires an allocated Slurm GPU"
    runtime_path = output / f"{stage}-runtime.json"
    runtime = {
        "stage": stage,
        "status": "running",
        "started_at": now(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "manifest_sha256": digest(manifest_path),
        "steps": [],
    }
    with runtime_path.open("x") as stream:
        stream.write(json.dumps(runtime, indent=2))
    env = os.environ.copy()
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_NAME"):
        env.pop(key, None)
    env.update(
        PYTHONPATH=str(source / "src"),
        MUJOCO_GL="egl",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        WANDB_RUN_GROUP=f"plug-recipe-{runtime['job_id']}",
        PYTHONUNBUFFERED="1",
        XDG_CACHE_HOME=f"/tmp/plug-recipe-{runtime['job_id']}",
    )

    def run(command):
        for relative, expected in plan["source_files_sha256"].items():
            assert digest(source / relative) == expected, relative
        step = {"command": command, "started_at": now()}
        runtime["steps"].append(step)
        save(runtime_path, runtime)
        print(json.dumps(step), flush=True)
        result = subprocess.run(command, cwd=source, env=env)
        step.update(returncode=result.returncode, finished_at=now())
        save(runtime_path, runtime)
        result.check_returncode()

    try:
        if stage == "smoke":
            run(
                [
                    plan["python"],
                    str(source / "validate_plug_repair_gpu.py"),
                    "--output",
                    str(output / "gpu-validation.json"),
                ]
            )
            assert json.loads((output / "gpu-validation.json").read_text())["passed"]
        else:
            assert json.loads((output / "gpu-validation.json").read_text())["passed"], (
                "GPU smoke must pass first"
            )
            for case in plan["cases"]:
                directory = output / case["name"]
                directory.mkdir(exist_ok=False)
                command = case["command"]
                log_root = Path(command[command.index("--log-root") + 1])
                log_root.mkdir(parents=True, exist_ok=True)
                before = set(log_root.iterdir())
                run(command)
                created = set(log_root.iterdir()) - before
                candidates = [p for p in created if case["name"] in p.name and p.is_dir()]
                assert len(candidates) == 1, candidates
                run_dir = candidates[0]
                checkpoint = run_dir / "model_750.pt"
                checkpoint_hash = digest(checkpoint)
                history = [
                    json.loads(line)
                    for line in (run_dir / "iterations.jsonl").read_text().splitlines()
                ]
                assert len(history) == 751, len(history)
                record = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "run_directory": str(run_dir),
                    "updates": 751,
                    "transitions": case["transitions"],
                    "evaluations": [],
                    "competence_gate_passed": True,
                }
                for repeat in (1, 2):
                    report = directory / f"evaluation-repeat{repeat}.json"
                    evaluation = [
                        plan["python"],
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
                        str(report),
                        "--wandb",
                    ]
                    if repeat == 1:
                        evaluation += ["--capture-representatives", str(directory / "traces")]
                    else:
                        evaluation += [
                            "--reference-report",
                            str(directory / "evaluation-repeat1.json"),
                        ]
                    run(evaluation)
                    measured = json.loads(report.read_text())
                    assert measured["episodes"] == 512 and digest(checkpoint) == checkpoint_hash
                    assert len(measured["per_variant"]) == 4
                    audit = measured["termination_audit"]
                    for key in (
                        "success_hold_violations",
                        "success_distance_violations",
                        "success_three_sample_violations",
                        "success_qpos_above_distance_threshold",
                    ):
                        assert audit[key] == 0, (key, audit[key])
                    if case["prepared"]["experiment"]["plug_wrist_yaw"]:
                        assert audit["success_orientation_violations"] == 0
                    for row in measured["per_variant"].values():
                        assert row["episodes"] == 128
                        record["competence_gate_passed"] &= row["successes"] >= 103
                    record["evaluations"].append(
                        {
                            "repeat": repeat,
                            "path": str(report),
                            "sha256": digest(report),
                            "successes": measured["successes"],
                            "per_variant": measured["per_variant"],
                        }
                    )
                    save(directory / "results.json", record)
                run(
                    [
                        plan["python"],
                        "-m",
                        "active_perception_arms.record_evaluation",
                        str(directory / "evaluation-repeat1.json"),
                        "--output",
                        str(directory / "videos"),
                        "--wandb",
                    ]
                )
        runtime.update(status="complete", finished_at=now())
    except BaseException as error:
        runtime.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
        raise
    finally:
        save(runtime_path, runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "smoke", "train"), required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(Path(__file__).resolve().parents[1], args.manifest.resolve())
    else:
        execute(args.manifest.resolve(), args.stage)


if __name__ == "__main__":
    main()
