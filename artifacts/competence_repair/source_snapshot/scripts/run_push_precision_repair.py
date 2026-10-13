"""Run a bounded fresh-optimizer repair with explicit push environment revisions."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--std", type=float, default=0.3)
    p.add_argument("--horizon", type=float, default=12)
    p.add_argument("--clear", action="store_true")
    p.add_argument("--wide", action="store_true")
    p.add_argument("--mean-penalty", type=float, default=0.0)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    root = Path(__file__).resolve().parents[1]
    state = json.loads(
        (root / "artifacts/push_rgb_arc120/runs/push_wrist_static_s2/state.json").read_text()
    )
    command = [
        os.sys.executable,
        "-m",
        "active_perception_arms.train",
        "--task",
        "push",
        "--task-revision",
        "push_v2",
        "--condition",
        "wrist_static",
        "--occlusion",
        "clean",
        "--seed",
        "2",
        "--num-envs",
        "2048",
        "--iterations",
        str(args.steps),
        "--episode-seconds",
        str(args.horizon),
        "--push-visibility-revision",
        "crossbar_v3",
        "--push-target-layout",
        "arc_v2",
        "--manip-control",
        "cartesian",
        "--reset-xy-jitter",
        ".02",
        "--reward-profile",
        "push_dense",
        "--initialize-from",
        state["checkpoint"],
        "--initial-std",
        str(args.std),
        "--entropy-coef",
        "0",
        "--learning-rate",
        ".0001",
        "--lr-schedule",
        "fixed",
        "--log-root",
        str(args.output / "logs"),
        "--result",
        str(args.output / "training.json"),
        "--save-interval",
        "250",
        "--run-label",
        "precision_repair",
    ]
    if args.mean_penalty:
        command += ["--action-mean-penalty", str(args.mean_penalty)]
    if args.clear:
        command += ["--push-workspace-revision", "arc_clear_v2"]
    if args.wide:
        command += ["--push-control-revision", "wide_v2"]
    (args.output / "command.json").write_text(json.dumps(command, indent=2))
    subprocess.run(command, check=True)
    checkpoints = sorted(
        (args.output / "logs").glob("*/model_*.pt"), key=lambda p: int(p.stem.split("_")[-1])
    )
    for checkpoint in checkpoints:
        cmd = [
            os.sys.executable,
            "-m",
            "active_perception_arms.evaluate",
            str(checkpoint),
            "--episodes",
            "512",
            "--num-envs",
            "128",
            "--seed",
            "31000",
            "--split",
            "validation",
            "--occlusion",
            "clean",
            "--audit-termination",
            "--output",
            str(args.output / f"evaluation-{checkpoint.stem}.json"),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
