"""Bounded state-only learnability diagnostic; never launches RGB experiments."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def breakdown(report):
    task = report["experiment"]["task"]
    counts = {}
    for row in report["termination_audit"]["records"]:
        if row["success"]:
            label = "success"
        elif task == "transfer":
            label = next(
                (
                    name
                    for name, field in (
                        ("no_lift", "transfer_lifted"),
                        ("not_placed", "transfer_placed"),
                        ("not_released", "transfer_released"),
                        ("not_settled", "transfer_settled"),
                    )
                    if not row[field]
                ),
                "not_withdrawn_or_held",
            )
        else:
            label = next(
                (
                    name
                    for name, failed in (
                        (
                            "outside_goal",
                            row["push_xy_error"] >= 0.015 or row["push_z_error"] >= 0.004,
                        ),
                        ("not_withdrawn", not row["push_withdrawn"]),
                        ("contact", not row["push_contact_free"]),
                        ("moving", row["linear_speed"] >= 0.035),
                    )
                    if failed
                ),
                "hold_not_completed",
            )
        counts[label] = counts.get(label, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--task", choices=("transfer", "push"), required=True)
    p.add_argument("--resume-diagnostic", action="store_true")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    source, root = Path(plan["source"]), Path(plan["root"])
    for name, digest in plan["source_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    output = Path(plan["output"]) / args.task
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "runtime.json"
    lock = (output / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if state_path.exists() and not args.resume_diagnostic:
        raise FileExistsError("Use a distinct diagnostic plan; do not repeat completed learning")
    state = dict(
        status="starting",
        task=args.task,
        job_id=os.environ.get("SLURM_JOB_ID"),
        updates=0,
        transitions=0,
        evaluations=[],
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
    if args.resume_diagnostic:
        state = json.loads(state_path.read_text())
        assert state["task"] == args.task
        if state.get("checkpoint"):
            assert (
                hashlib.sha256(Path(state["checkpoint"]).read_bytes()).hexdigest()
                == state["checkpoint_sha256"]
            )
        state.setdefault("resumptions", []).append(
            dict(
                job_id=os.environ.get("SLURM_JOB_ID"),
                previous_status=state["status"],
                previous_error=state.pop("error", None),
                resumed_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
        state["job_id"] = os.environ.get("SLURM_JOB_ID")
    save(state_path, state)
    env = dict(
        os.environ,
        PYTHONPATH=str(source / "src"),
        MUJOCO_GL="egl",
        WANDB_RUN_GROUP=plan["label"],
        WANDB_JOB_TYPE="diagnostic",
        WANDB_TAGS="privileged,learnability," + args.task,
    )

    def report():
        with (Path(plan["output"]) / "report.lock").open("w") as report_lock:
            fcntl.flock(report_lock, fcntl.LOCK_EX)
            subprocess.run(
                [
                    plan["python"],
                    str(root / "scripts/build_privileged_report.py"),
                    "--root",
                    str(root),
                ],
                cwd=source,
                env=env,
                check=True,
            )

    def execute(command, status):
        state.update(status=status, current_command=command)
        save(state_path, state)
        with (output / "commands.jsonl").open("a") as log:
            log.write(json.dumps(command) + "\n")
        subprocess.run(command, cwd=source, env=env, check=True)

    checkpoint = Path(state["checkpoint"]) if state.get("checkpoint") else None
    try:
        for budget in plan["milestones"]:
            updates = math.ceil(budget / (plan["num_envs"] * 24))
            command = [
                plan["python"],
                "-m",
                "active_perception_arms.train",
                "--control-revision",
                "legacy_v1",
                "--task",
                args.task,
                "--condition",
                "wrist_static",
                "--fixed-view",
                "4" if args.task == "transfer" else "0",
                "--actor-observation",
                "privileged",
                "--memory",
                "none",
                "--manip-control",
                "cartesian",
                "--task-revision",
                "pick_place_v2" if args.task == "transfer" else "push_v2",
                "--reward-profile",
                args.task + "_dense",
                "--occlusion",
                "dynamic",
                "--reset-xy-jitter",
                "0.02",
                "--num-envs",
                str(plan["num_envs"]),
                "--seed",
                "0",
                "--iterations",
                str(updates),
                "--initial-std",
                "1",
                "--entropy-coef",
                "0.01",
                "--learning-rate",
                "0.0003",
                "--lr-schedule",
                "adaptive",
                "--save-interval",
                "25",
                "--job-type",
                "pilot",
                "--run-label",
                plan["label"],
                "--log-root",
                str(Path(plan["logs"]) / args.task),
                "--result",
                str(output / f"training-{updates}.json"),
            ]
            if checkpoint:
                command += ["--resume", str(checkpoint)]
            if state["updates"] < updates:
                execute(command, "training")
            elif state["updates"] > updates:
                continue
            dirs = list((Path(plan["logs"]) / args.task).glob("*/experiment.json"))
            assert len(dirs) == 1
            checkpoint = dirs[0].parent / f"model_{updates - 1}.pt"
            import torch

            assert (
                torch.load(checkpoint, map_location="cpu", weights_only=False)["iter"] + 1
                == updates
            )
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            state.update(
                updates=updates,
                transitions=updates * plan["num_envs"] * 24,
                checkpoint=str(checkpoint),
                checkpoint_sha256=digest,
                run_directory=str(checkpoint.parent),
            )
            save(state_path, state)
            for seed in (30000, 31000):
                path = output / f"validation-{updates}-s{seed}.json"
                if any(r["path"] == str(path) for r in state["evaluations"]):
                    continue
                command = [
                    plan["python"],
                    "-m",
                    "active_perception_arms.evaluate",
                    str(checkpoint),
                    "--episodes",
                    "512",
                    "--num-envs",
                    "128",
                    "--seed",
                    str(seed),
                    "--split",
                    "validation",
                    "--audit-termination",
                    "--output",
                    str(path),
                ]
                if budget == plan["milestones"][-1] and seed == 30000:
                    command += ["--capture-representatives", str(output / f"capture-{updates}")]
                execute(command, "evaluating")
                result = json.loads(path.read_text())
                audit = result["termination_audit"]
                assert result["episodes"] == 512 and result["successes"] == sum(result["outcomes"])
                assert (
                    audit["success_hold_violations"]
                    == audit["success_" + args.task + "_violations"]
                    == 0
                )
                assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
                state["evaluations"].append(
                    dict(
                        path=str(path),
                        transitions=state["transitions"],
                        seed=seed,
                        successes=result["successes"],
                        episodes=512,
                        success_rate=result["success_rate"],
                        stages=breakdown(result),
                    )
                )
                save(state_path, state)
                report()
                if result.get("representative_capture"):
                    execute(
                        [
                            plan["python"],
                            str(source / "scripts/render_transfer_capture.py"),
                            str(path),
                            str(output / f"videos-{updates}"),
                        ],
                        "rendering",
                    )
        state["status"] = "completed_diagnostic"
        last = state["evaluations"][-2:]
        state["competence_demonstrated"] = all(e["success_rate"] >= 0.9 for e in last)
        state["next_step"] = "Review state-policy results and trajectories before any RGB training."
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save(state_path, state)
        report()


if __name__ == "__main__":
    main()
