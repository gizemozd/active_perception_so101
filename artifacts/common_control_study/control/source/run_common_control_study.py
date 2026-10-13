"""Prepare and execute bounded, hash-pinned plug control/camera comparisons."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if (output / "plan.json").exists():
        raise FileExistsError("Use the existing plan; do not duplicate the study")
    snapshot = output / "source"
    shutil.copytree(
        root / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    shutil.copy2(
        root / "scripts/validate_common_control_gpu.py", snapshot / "validate_common_control_gpu.py"
    )
    hashes = {
        str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(snapshot.rglob("*"))
        if p.is_file()
    }
    cases = []
    recipes = (
        ["absolute", "delta"]
        if args.stage == "control"
        else ["bounded", "bounded_slow", "bounded_dynamic"]
    )
    for seed in (0, 1):
        for condition in ("wrist", "wrist_static", "active"):
            for recipe in recipes:
                if recipe == "bounded_slow" and condition != "active":
                    continue
                name = f"{recipe}_{condition}_s{seed}"
                case_dir = output / name
                command = [
                    str(root / ".venv/bin/python"),
                    "-m",
                    "active_perception_arms.train",
                    "--task",
                    "plug",
                    "--condition",
                    condition,
                    "--control-revision",
                    "pose_v1",
                    "--translation-control",
                    "absolute" if recipe == "absolute" else "delta",
                    "--translation-step",
                    "0.005",
                    "--camera-interval",
                    "4" if recipe == "bounded_slow" else "1",
                    "--seed",
                    str(seed),
                    "--num-envs",
                    "256",
                    "--width",
                    "128",
                    "--height",
                    "96",
                    "--iterations",
                    str(args.iterations),
                    "--initial-seconds",
                    "1",
                    "--episode-seconds",
                    "3.5",
                    "--reward-profile",
                    "plug_insertion",
                    "--memory",
                    "gru",
                    "--occlusion",
                    "dynamic" if recipe == "bounded_dynamic" else "clean",
                    "--initial-std",
                    "1" if args.stage == "control" else "0.3",
                    "--entropy-coef",
                    "0.01" if args.stage == "control" else "0.003",
                    "--action-distribution",
                    "gaussian" if args.stage == "control" else "squashed",
                    "--learning-rate",
                    "0.0003",
                    "--lr-schedule",
                    "adaptive",
                    "--save-interval",
                    str(max(1, args.iterations // 2)),
                    "--log-root",
                    str(case_dir / "logs"),
                    "--result",
                    str(case_dir / "training.json"),
                    "--run-label",
                    name,
                ]
                dry = json.loads(
                    subprocess.check_output(
                        command + ["--dry-run"],
                        env=os.environ | {"PYTHONPATH": str(snapshot / "src")},
                        text=True,
                    )
                )
                cases.append({"name": name, "command": command, "prepared": dry})
    # Compare every absolute/delta pair, with no hidden reward/sensor/optimizer changes.
    if args.stage == "control":
        for i in range(0, len(cases), 2):
            a, b = [c["prepared"] for c in cases[i : i + 2]]
            diffs = {k for k in a["experiment"] if a["experiment"][k] != b["experiment"][k]}
            assert diffs == {"translation_control", "plug_control"}, diffs
            assert {k: v for k, v in a.items() if k != "experiment"} == {
                k: v for k, v in b.items() if k != "experiment"
            }
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "output": str(output),
        "source": str(snapshot),
        "source_sha256": hashes,
        "stage": args.stage,
        "cases": cases,
        "transitions_per_policy": args.iterations * 256 * 24,
        "max_concurrent_gpus": 4,
        "selection": "Highest clean validation success, earliest checkpoint breaks ties; final tests never select checkpoints.",
        "validation_seed": 110000,
        "test_seed": 120000,
        "scope": "Original four-variant plug; identical home-camera calibration, physical scoring, reward and timing; small training budget is a screen, not a convergence claim.",
    }
    save(output / "plan.json", plan)
    print(
        json.dumps(
            {
                "plan": str(output / "plan.json"),
                "cases": len(cases),
                "transitions_each": plan["transitions_per_policy"],
            }
        )
    )


def run(args):
    plan = json.loads(args.plan.read_text())
    snapshot = Path(plan["source"])
    for name, expected in plan["source_sha256"].items():
        assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == expected, name
    case = plan["cases"][args.index]
    output = Path(plan["output"]) / case["name"]
    if (output / "completed.json").exists():
        print("Already complete:", case["name"])
        return
    env = os.environ | {
        "PYTHONPATH": str(snapshot / "src"),
        "WANDB_RUN_GROUP": "common-controls-20261012-" + plan["stage"],
    }
    output.mkdir(parents=True, exist_ok=True)
    save(
        output / "started.json", {"job": os.environ.get("SLURM_JOB_ID"), "command": case["command"]}
    )

    def call(cmd, name):
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(
                cmd, cwd=plan["root"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    if not (output / "training.json").exists():
        call(case["command"], "training")
    checkpoints = sorted(
        (output / "logs").glob("*/model_*.pt"), key=lambda p: int(p.stem.split("_")[-1])
    )
    evaluated = []
    for ckpt in checkpoints:
        if ckpt.stem == "model_0":
            continue
        result = output / f"validation-{ckpt.stem}.json"
        cmd = [
            sys.executable,
            "-m",
            "active_perception_arms.evaluate",
            str(ckpt),
            "--episodes",
            "256",
            "--num-envs",
            "128",
            "--seed",
            str(plan["validation_seed"]),
            "--split",
            "validation",
            "--occlusion",
            "clean",
            "--audit-termination",
            "--output",
            str(result),
        ]
        if not result.exists():
            call(cmd, result.stem)
        evaluated.append((json.loads(result.read_text())["successes"], ckpt))
    best = max(evaluated, key=lambda p: (p[0], -int(p[1].stem.split("_")[-1])))[1]
    save(
        output / "selection.json",
        {
            "checkpoint": str(best),
            "validation_successes": max(p[0] for p in evaluated),
            "rule": plan["selection"],
        },
    )
    for mode in ("clean", "dynamic"):
        result = output / f"test-{mode}.json"
        cmd = [
            sys.executable,
            "-m",
            "active_perception_arms.evaluate",
            str(best),
            "--episodes",
            "512",
            "--num-envs",
            "128",
            "--seed",
            str(plan["test_seed"]),
            "--split",
            "test",
            "--occlusion",
            mode,
            "--audit-termination",
            "--output",
            str(result),
        ]
        if not result.exists():
            call(cmd, result.stem)
    save(
        output / "completed.json",
        {
            "selected_checkpoint": str(best),
            "source_manifest": str(args.plan),
            "results": {
                mode: json.loads((output / f"test-{mode}.json").read_text())["success_rate"]
                for mode in ("clean", "dynamic")
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--stage", choices=("control", "camera"), default="control")
    prep.add_argument("--iterations", type=int, default=200)
    execute = sub.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--index", type=int, required=True)
    args = p.parse_args()
    prepare(args) if args.action == "prepare" else run(args)
