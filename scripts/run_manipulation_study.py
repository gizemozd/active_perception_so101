"""Source-pinned, resumable billion-transition transfer/push study.

Each policy has its own immutable settings, checkpoints and evaluation records.
Seed-zero static learning gates expansion within a task. Gates never shorten the
one-billion-transition commitment of a started policy. Slurm wall-time signals
checkpoint the learner and submit a dependency-linked continuation.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def study_suspended(plan):
    suspension = Path(plan["output"]) / "suspension.json"
    if not suspension.exists():
        return False
    state = json.loads(suspension.read_text())
    return state.get("retired", False) or not state.get("automatic_expansion_allowed", False)


def submit(plan_path, run_id, *, continuation=False):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if study_suspended(plan):
        raise RuntimeError("RGB study stopped for privileged validation; submission is disabled")
    folder = Path(plan["output"]) / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        attempts = sorted(folder.glob("submission-*.json"))
        if attempts and not continuation:
            return json.loads(attempts[0].read_text())
        if len(attempts) >= 8:
            raise RuntimeError("Eight allocations exhausted; inspect before another continuation")
        reservation = folder / f"submission-{len(attempts):02}.json"
        env = dict(os.environ)
        env.pop("SLURM_STEPMGR", None)
        command = [
            "sbatch",
            "--parsable",
            "--export=ALL",
            "--account=kempner_pgozdil_lab",
            "--partition=kempner_rtx",
            "--exclude=holygpu7c2313",
            "--chdir=" + plan["root"],
            "--job-name=mp-" + run_id,
        ]
        if continuation and os.environ.get("SLURM_JOB_ID"):
            command += ["--dependency=afterany:" + os.environ["SLURM_JOB_ID"]]
        command += [
            str(Path(plan["source"]) / "scripts/slurm/manipulation_study.sbatch"),
            plan["python"],
            str(Path(plan["source"]) / "scripts/run_manipulation_study.py"),
            str(plan_path),
            run_id,
        ]
        record = dict(status="submitting", command=command, created_utc=now())
        save(reservation, record)
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        record["status"] = "submitted" if result.returncode == 0 else "submission_failed"
        if result.returncode == 0:
            record["job_id"] = result.stdout.strip().split(";")[0]
        save(reservation, record)
        result.check_returncode()
        return record


def verify_source(plan):
    for relative, expected in plan["source_sha256"].items():
        if digest(Path(plan["source"]) / relative) != expected:
            raise RuntimeError("Frozen source changed: " + relative)
    if digest(plan["validation"]) != plan["validation_sha256"]:
        raise RuntimeError("GPU validation changed")
    validation = json.loads(Path(plan["validation"]).read_text())
    for task in ("transfer", "push"):
        assert validation[task]["smoke"]["passed"]
        assert validation[task]["successes"] / validation[task]["episodes"] >= 0.95


def command_for(plan, run, total_updates, checkpoint=None):
    command = [
        plan["python"],
        "-m",
        "active_perception_arms.train",
        "--control-revision",
        "legacy_v1",
        "--task",
        run["task"],
        "--condition",
        run["condition"],
        "--seed",
        str(run["seed"]),
        "--num-envs",
        str(plan["num_envs"]),
        "--iterations",
        str(total_updates),
        "--occlusion",
        "dynamic",
        "--manip-control",
        "cartesian",
        "--task-revision",
        "pick_place_v2" if run["task"] == "transfer" else "push_v2",
        "--reward-profile",
        run["task"] + "_dense",
        "--reset-xy-jitter",
        str(plan["jitter"]),
        "--initial-std",
        "1.0",
        "--learning-rate",
        "0.0003",
        "--lr-schedule",
        "adaptive",
        "--entropy-coef",
        "0.01",
        "--memory",
        "gru",
        "--save-interval",
        "1000",
        "--fixed-view",
        str(run["fixed_view"]),
        "--job-type",
        "study",
        "--run-label",
        plan["label"],
        "--log-root",
        str(Path(plan["logs"]) / run["id"]),
        "--result",
        str(Path(plan["output"]) / "runs" / run["id"] / "training.json"),
    ]
    if checkpoint:
        command += ["--resume", str(checkpoint)]
    return command


def build_report(plan):
    # One writer across all GPU workers; report snapshots are read-only evidence consumers.
    with (Path(plan["output"]) / "report.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(
            [
                plan["python"],
                str(Path(plan["source"]) / "scripts/build_manipulation_report.py"),
                "--root",
                plan["root"],
            ],
            check=True,
        )


def work(plan_path, run_id):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if study_suspended(plan):
        raise RuntimeError("RGB study stopped for privileged validation; worker is disabled")
    verify_source(plan)
    run = next(r for r in plan["runs"] if r["id"] == run_id)
    folder = Path(plan["output"]) / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = folder / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else dict(
            run=run,
            status="starting",
            evaluations=[],
            updates=0,
            transitions=0,
            allocations=[],
        )
    )
    state["allocations"].append(dict(job_id=os.environ.get("SLURM_JOB_ID"), started_utc=now()))
    state["status"] = "running"
    save(state_path, state)
    os.environ.update(
        PYTHONPATH=str(Path(plan["source"]) / "src"),
        MUJOCO_GL="egl",
        WANDB_RUN_GROUP=plan["label"],
        WANDB_TAGS="transfer-push,billion," + run["task"] + "," + run["condition"],
        WANDB_JOB_TYPE="study",
    )
    stopping = []
    child = None

    def stop(signum, frame):
        stopping.append(signum)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGUSR1 if state["status"] == "training" else signal.SIGTERM)

    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)

    def execute(command, status):
        nonlocal child
        state.update(status=status, updated_utc=now(), current_command=command)
        save(state_path, state)
        with (folder / "commands.jsonl").open("a") as log:
            log.write(json.dumps(dict(time=now(), command=command)) + "\n")
        child = subprocess.Popen(command, cwd=plan["source"])
        returncode = child.wait()
        child = None
        if returncode != 0 and not stopping:
            raise subprocess.CalledProcessError(returncode, command)

    checkpoint = Path(state["checkpoint"]) if state.get("checkpoint") else None
    batch = plan["num_envs"] * 24
    final_updates = math.ceil(plan["min_transitions"] / batch)
    milestones = sorted(
        set(
            [math.ceil(v / batch) for v in (12_000_000, 50_000_000, 100_000_000)]
            + list(
                range(math.ceil(200_000_000 / batch), final_updates, math.ceil(100_000_000 / batch))
            )
            + [final_updates]
        )
    )
    try:
        for updates in milestones:
            if stopping:
                break
            if updates > state["updates"]:
                verify_source(plan)
                execute(command_for(plan, run, updates, checkpoint), "training")
                dirs = list((Path(plan["logs"]) / run_id).glob("*"))
                dirs = [p for p in dirs if (p / "experiment.json").exists()]
                assert len(dirs) == 1, dirs
                run_dir = dirs[0]
                checkpoint = run_dir / ("interrupted.pt" if stopping else f"model_{updates - 1}.pt")
                import torch

                actual = (
                    int(torch.load(checkpoint, map_location="cpu", weights_only=False)["iter"]) + 1
                )
                if not stopping:
                    assert actual == updates
                state.update(
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=digest(checkpoint),
                    updates=actual,
                    transitions=actual * batch,
                    run_directory=str(run_dir),
                )
                save(state_path, state)
                if stopping:
                    break
            # A checkpoint reached before interruption may still need its evaluations.
            if updates != state["updates"]:
                continue
            final = updates == final_updates
            split, seed = ("test", 80000) if final else ("validation", 30000)
            for occlusion in ("clean", "dynamic"):
                path = folder / f"evaluation-{updates}-{occlusion}-{split}.json"
                if not path.exists():
                    command = [
                        plan["python"],
                        "-m",
                        "active_perception_arms.evaluate",
                        str(checkpoint),
                        "--episodes",
                        "1024" if final else "512",
                        "--num-envs",
                        "128",
                        "--seed",
                        str(seed),
                        "--split",
                        split,
                        "--occlusion",
                        occlusion,
                        "--audit-termination",
                        "--output",
                        str(path),
                    ]
                    if final:
                        command += [
                            "--capture-representatives",
                            str(folder / ("traces-" + occlusion)),
                        ]
                    execute(command, "evaluating")
                if stopping:
                    break
                report = json.loads(path.read_text())
                audit = report["termination_audit"]
                assert report["successes"] == sum(report["outcomes"])
                assert report["episodes"] == (1024 if final else 512)
                assert (
                    audit["success_hold_violations"]
                    == audit["success_" + run["task"] + "_violations"]
                    == 0
                )
                assert digest(checkpoint) == state["checkpoint_sha256"]
                record = dict(
                    path=str(path),
                    transitions=updates * batch,
                    split=split,
                    occlusion=occlusion,
                    successes=report["successes"],
                    episodes=report["episodes"],
                    success_rate=report["success_rate"],
                    sha256=digest(path),
                )
                state["evaluations"] = [
                    r for r in state["evaluations"] if r["path"] != str(path)
                ] + [record]
                save(state_path, state)
                if final:
                    execute(
                        [
                            plan["python"],
                            str(Path(plan["source"]) / "scripts/render_transfer_capture.py"),
                            str(path),
                            str(folder / ("videos-" + occlusion)),
                        ],
                        "rendering",
                    )
            if stopping:
                break
            recent = [
                r
                for r in state["evaluations"]
                if r["transitions"] == updates * batch and r["split"] == "validation"
            ]
            if run["condition"] == "wrist_static" and run["seed"] == 0:
                if len(recent) == 2 and all(
                    r["success_rate"] >= plan["competence_gate"] for r in recent
                ):
                    state["competence_gate"] = dict(
                        passed=True, transitions=updates * batch, evaluations=recent
                    )
                    save(state_path, state)
                    for other in plan["runs"]:
                        if other["task"] == run["task"] and other["id"] != run_id:
                            submit(plan_path, other["id"])
            build_report(plan)
        if not stopping and run["condition"] == "active":
            for label, flag in (
                ("freeze_camera", "--freeze-camera-after"),
                ("hold_external", "--hold-external-after"),
            ):
                path = folder / f"intervention-{label}.json"
                if not path.exists():
                    execute(
                        [
                            plan["python"],
                            "-m",
                            "active_perception_arms.evaluate",
                            str(checkpoint),
                            "--episodes",
                            "1024",
                            "--num-envs",
                            "128",
                            "--seed",
                            "80000",
                            "--split",
                            "test",
                            "--occlusion",
                            "dynamic",
                            flag,
                            "1.0",
                            "--audit-termination",
                            "--output",
                            str(path),
                        ],
                        "evaluating",
                    )
                if stopping:
                    break
                report = json.loads(path.read_text())
                assert report["successes"] == sum(report["outcomes"])
                assert report["termination_audit"]["success_" + run["task"] + "_violations"] == 0
                state.setdefault("interventions", {})[label] = dict(
                    path=str(path),
                    successes=report["successes"],
                    episodes=report["episodes"],
                    success_rate=report["success_rate"],
                    sha256=digest(path),
                )
                save(state_path, state)
        if stopping:
            # Only the explicit wall-time warning may schedule another allocation.
            # Slurm cancellation sends TERM even when scancel requests KILL here.
            if signal.SIGTERM in stopping or study_suspended(plan):
                state["status"] = "stopped"
                state.pop("continuation", None)
            else:
                state["status"] = "checkpointed_for_continuation"
                save(state_path, state)
                state["continuation"] = submit(plan_path, run_id, continuation=True)
        else:
            assert state["transitions"] >= plan["min_transitions"]
            state["status"] = "completed"
        state["updated_utc"] = now()
        save(state_path, state)
        build_report(plan)
    except BaseException as error:
        state.update(status="failed", error=repr(error), updated_utc=now())
        save(state_path, state)
        build_report(plan)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    if args.submit:
        print(json.dumps(submit(args.plan, args.run), indent=2))
    else:
        work(args.plan, args.run)


if __name__ == "__main__":
    main()
