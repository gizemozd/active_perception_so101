#!/usr/bin/env python3
"""Prepare and run matched 18M plug insertion-reward camera policies."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def command_for(plan, case, updates, checkpoint=None):
    cmd = [
        plan["python"],
        "-m",
        "active_perception_arms.train",
        "--control-revision",
        "legacy_v1",
        "--task",
        "plug",
        "--condition",
        case["condition"],
        "--seed",
        str(plan["training_seed"]),
        "--num-envs",
        str(plan["num_envs"]),
        "--iterations",
        str(updates),
        "--occlusion",
        "clean",
        "--plug-control",
        "absolute",
        "--reward-profile",
        "plug_insertion",
        "--no-plug-wrist-yaw",
        "--memory",
        "gru",
        "--policy-profile",
        "compact",
        "--initial-std",
        "1.0",
        "--learning-rate",
        "0.0003",
        "--lr-schedule",
        "adaptive",
        "--entropy-coef",
        "0.01",
        "--fixed-view",
        str(case["fixed_view"]),
        "--save-interval",
        "250",
        "--job-type",
        "study",
        "--run-label",
        f"plug_insertion_18m_seed{plan['training_seed']}",
        "--log-root",
        str(Path(plan["logs"]) / case["condition"]),
        "--result",
        str(Path(plan["output"]) / case["condition"] / "training.json"),
    ]
    if checkpoint:
        cmd += ["--resume", str(checkpoint)]
    return cmd


def prepare(root, path, seed=0):
    if path.exists():
        raise FileExistsError(path)
    output = path.parent
    output.mkdir(parents=True, exist_ok=True)
    pilot_plan_path = root / "artifacts/plug_reward_pilot/20261009T230424Z/plan.json"
    pilot_summary_path = pilot_plan_path.with_name("summary.json")
    pilot_plan = json.loads(pilot_plan_path.read_text())
    pilot_summary = json.loads(pilot_summary_path.read_text())
    assert pilot_summary["updates"] == 100
    assert pilot_summary["transitions"] == 1_228_800
    assert pilot_summary["reward_increased"]
    assert pilot_summary["evaluations"][-1]["successes"] == 2
    assert pilot_summary["evaluations"][-1]["physical_success_violations"] == 0
    source = (
        root
        / ".cache"
        / (f"plug-insertion-study-seed{seed}-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    shutil.copytree(
        Path(pilot_plan["source"]) / "src",
        source / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for relative, expected in pilot_plan["source_sha256"].items():
        if relative.startswith("src/"):
            assert digest(source / relative) == expected, relative
    for name in ("run_plug_insertion_study.py", "build_plug_insertion_study_report.py"):
        shutil.copy2(root / "scripts" / name, source / name)
    shutil.copy2(root / "scripts/slurm/plug_precision.sbatch", source / "run.sbatch")
    plan = dict(
        created_at=now(),
        root=str(root),
        source=str(source),
        output=str(output),
        logs=str(root / "logs/plug_insertion_study" / f"seed_{seed}"),
        python=str(root / ".venv/bin/python"),
        source_provenance=dict(
            pilot_plan=str(pilot_plan_path),
            pilot_summary=str(pilot_summary_path),
            pilot_summary_sha256=digest(pilot_summary_path),
            source_note="Training source copied byte-for-byte from the completed 100-update GPU pilot.",
        ),
        num_envs=512,
        requested_transitions_per_policy=18_000_000,
        updates=math.ceil(18_000_000 / (512 * 24)),
        milestones=[750, math.ceil(18_000_000 / (512 * 24))],
        training_seed=seed,
        evaluation=dict(
            validation_episodes=512,
            validation_seed=30000,
            final_episodes=1024,
            final_seed=80000,
            settings=["clean", "dynamic"],
        ),
        cases=[],
    )
    plan["transitions_per_policy"] = plan["updates"] * plan["num_envs"] * 24
    env = os.environ | {"PYTHONPATH": str(source / "src")}
    for condition in ("wrist", "wrist_static", "active"):
        case = dict(condition=condition, fixed_view=7 if condition == "wrist_static" else 0)
        command = command_for(plan, case, plan["updates"])
        prepared = json.loads(
            subprocess.check_output(command + ["--dry-run"], cwd=source, env=env, text=True)
        )
        experiment = prepared["experiment"]
        assert experiment["actor_observation"] == "rgb"
        assert experiment["reward_profile"] == "plug_insertion"
        assert experiment["plug_control"] == "absolute"
        assert experiment["success_state_sample"] == "current_qpos"
        assert prepared["initial_std"] == 1.0
        assert prepared["optimization"]["entropy_coef"] == 0.01
        case.update(command=command, prepared=prepared)
        plan["cases"].append(case)
    plan["total_training_transitions"] = len(plan["cases"]) * plan["transitions_per_policy"]
    plan["source_sha256"] = {
        str(p.relative_to(source)): digest(p)
        for p in sorted(source.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    save(path, plan)
    print(
        json.dumps(
            dict(
                plan=str(path),
                transitions_each=plan["transitions_per_policy"],
                total=plan["total_training_transitions"],
                source=str(source),
            ),
            indent=2,
        )
    )


def verify(plan):
    for relative, expected in plan["source_sha256"].items():
        assert digest(Path(plan["source"]) / relative) == expected, relative
    pilot = plan["source_provenance"]
    assert digest(pilot["pilot_summary"]) == pilot["pilot_summary_sha256"]


def submit(path, condition, continuation=False):
    plan = json.loads(path.read_text())
    verify(plan)
    assert condition in [c["condition"] for c in plan["cases"]]
    folder = Path(plan["output"]) / condition
    folder.mkdir(exist_ok=True)
    with (folder / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = sorted(folder.glob("submission-*.json"))
        if previous and not continuation:
            record = json.loads(previous[0].read_text())
            assert record["status"] == "submitted", record
            return record
        assert len(previous) < 3, "Three allocations exhausted; inspect before another continuation"
        target = folder / f"submission-{len(previous):02}.json"
        cmd = [
            "sbatch",
            "--parsable",
            "--account=kempner_pgozdil_lab",
            "--partition=kempner_rtx",
            "--exclude=holygpu7c2313",
            "--chdir=" + plan["root"],
            "--job-name=plug-insertion-" + condition,
        ]
        if continuation:
            cmd += ["--dependency=afterany:" + os.environ["SLURM_JOB_ID"]]
        cmd += [
            str(Path(plan["source"]) / "run.sbatch"),
            plan["python"],
            str(Path(plan["source"]) / "run_plug_insertion_study.py"),
            str(path),
            condition,
        ]
        record = dict(status="submitting", created_at=now(), command=cmd)
        save(target, record)
        env = os.environ.copy()
        env.pop("SLURM_STEPMGR", None)
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        record.update(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            status="submitted" if result.returncode == 0 else "submission_failed",
        )
        if result.returncode == 0:
            record["job_id"] = result.stdout.strip().split(";")[0]
        save(target, record)
        result.check_returncode()
        return record


def report(plan):
    with (Path(plan["output"]) / "report.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(
            [
                plan["python"],
                str(Path(plan["source"]) / "build_plug_insertion_study_report.py"),
                "--output",
                plan["output"],
            ],
            check=True,
        )


def work(path, condition):
    plan = json.loads(path.read_text())
    verify(plan)
    case = next(c for c in plan["cases"] if c["condition"] == condition)
    folder = Path(plan["output"]) / condition
    folder.mkdir(exist_ok=True)
    with (folder / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = folder / "state.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else dict(
                condition=condition,
                status="starting",
                updates=0,
                transitions=0,
                evaluations=[],
                allocations=[],
            )
        )
        state["allocations"].append(dict(job_id=os.environ.get("SLURM_JOB_ID"), started_at=now()))
        save(state_path, state)
        env = os.environ.copy()
        for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_NAME"):
            env.pop(key, None)
        env.update(
            PYTHONPATH=str(Path(plan["source"]) / "src"),
            MUJOCO_GL="egl",
            WANDB_RUN_GROUP="plug-insertion-reward-18m",
            WANDB_TAGS=f"plug,plug_insertion,18m,seed{plan['training_seed']},{condition}",
        )
        stopping, child = [], None

        def stop(signum, frame):
            stopping.append(signum)
            if child is not None and child.poll() is None:
                child.send_signal(
                    signal.SIGUSR1 if state["status"] == "training" else signal.SIGTERM
                )

        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, stop)

        def execute(cmd, status):
            nonlocal child
            verify(plan)
            state.update(status=status, current_command=cmd, updated_at=now())
            save(state_path, state)
            with (folder / "commands.jsonl").open("a") as log:
                log.write(json.dumps(dict(time=now(), command=cmd)) + "\n")
            child = subprocess.Popen(cmd, cwd=plan["source"], env=env)
            code = child.wait()
            child = None
            if code and not stopping:
                raise subprocess.CalledProcessError(code, cmd)

        checkpoint = Path(state["checkpoint"]) if state.get("checkpoint") else None
        try:
            for updates in plan["milestones"]:
                if stopping:
                    break
                if updates > state["updates"]:
                    execute(command_for(plan, case, updates, checkpoint), "training")
                    dirs = [
                        p
                        for p in (Path(plan["logs"]) / condition).glob("*")
                        if (p / "experiment.json").exists()
                    ]
                    assert len(dirs) == 1, dirs
                    run_dir = dirs[0]
                    checkpoint = run_dir / (
                        "interrupted.pt" if stopping else f"model_{updates - 1}.pt"
                    )
                    import torch

                    actual = (
                        int(torch.load(checkpoint, map_location="cpu", weights_only=True)["iter"])
                        + 1
                    )
                    assert stopping or actual == updates
                    state.update(
                        checkpoint=str(checkpoint),
                        checkpoint_sha256=digest(checkpoint),
                        updates=actual,
                        transitions=actual * plan["num_envs"] * 24,
                        run_directory=str(run_dir),
                    )
                    save(state_path, state)
                if stopping:
                    break
                if state["updates"] != updates:
                    continue
                final = updates == plan["updates"]
                split, seed, episodes = (
                    ("test", 80000, 1024) if final else ("validation", 30000, 512)
                )
                for setting in ("clean", "dynamic"):
                    output = folder / f"evaluation-{updates}-{setting}-{split}.json"
                    if not output.exists():
                        cmd = [
                            plan["python"],
                            "-m",
                            "active_perception_arms.evaluate",
                            str(checkpoint),
                            "--episodes",
                            str(episodes),
                            "--num-envs",
                            "128",
                            "--seed",
                            str(seed),
                            "--split",
                            split,
                            "--occlusion",
                            setting,
                            "--audit-termination",
                            "--output",
                            str(output),
                        ]
                        if final:
                            cmd += ["--capture-representatives", str(folder / f"traces-{setting}")]
                        execute(cmd, "evaluating")
                    if stopping:
                        break
                    data = json.loads(output.read_text())
                    assert (
                        data["episodes"] == episodes and sum(data["outcomes"]) == data["successes"]
                    )
                    for key in (
                        "success_hold_violations",
                        "success_three_sample_violations",
                        "success_qpos_above_distance_threshold",
                    ):
                        assert data["termination_audit"][key] == 0
                    assert digest(checkpoint) == state["checkpoint_sha256"]
                    record = dict(
                        path=str(output),
                        sha256=digest(output),
                        updates=updates,
                        transitions=updates * plan["num_envs"] * 24,
                        split=split,
                        occlusion=setting,
                        successes=data["successes"],
                        episodes=episodes,
                        per_variant=data["per_variant"],
                    )
                    state["evaluations"] = [
                        r for r in state["evaluations"] if r["path"] != str(output)
                    ] + [record]
                    save(state_path, state)
                    report(plan)
                    if final:
                        execute(
                            [
                                plan["python"],
                                "-m",
                                "active_perception_arms.record_evaluation",
                                str(output),
                                "--output",
                                str(folder / f"videos-{setting}"),
                            ],
                            "rendering",
                        )
                if stopping:
                    break
            if signal.SIGUSR1 in stopping:
                state["continuation"] = submit(path, condition, continuation=True)
                state["status"] = "checkpointed_for_continuation"
            elif stopping:
                state["status"] = "stopped"
            else:
                assert state["transitions"] == plan["transitions_per_policy"]
                state.update(status="completed", completed_at=now())
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save(state_path, state)
            report(plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--condition", choices=("wrist", "wrist_static", "active"))
    p.add_argument("--seed", type=int, default=0, help="Training seed for --prepare")
    args = p.parse_args()
    path = args.plan.resolve()
    if args.prepare:
        prepare(Path(__file__).resolve().parents[1], path, args.seed)
    elif args.submit:
        print(json.dumps(submit(path, args.condition), indent=2))
    else:
        work(path, args.condition)


if __name__ == "__main__":
    main()
