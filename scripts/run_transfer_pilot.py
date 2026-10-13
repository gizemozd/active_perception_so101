#!/usr/bin/env python3
"""Run one source-pinned transfer learning pilot and audit its final checkpoint."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root, source, output = (Path(plan[k]) for k in ("root", "source", "output"))
    for name, expected in plan["source_sha256"].items():
        assert digest(source / name) == expected, name
    assert digest(plan["validation"]) == plan["validation_sha256"]
    validation = json.loads(Path(plan["validation"]).read_text())
    assert validation["ppo_smoke"]["passed"]
    assert all(
        c["success_hold_violations"] == c["success_transfer_violations"] == 0
        for c in validation["cases"]
    )
    successes = sum(c["terminal_flags"]["success"] for c in validation["cases"])
    count = sum(c["episodes"] for c in validation["cases"])
    assert successes / count >= 0.95
    runtime = dict(
        status="running",
        job_id=os.environ.get("SLURM_JOB_ID"),
        steps=[],
        started_utc=datetime.now(timezone.utc).isoformat(),
        learned_success_rate=None,
    )
    state_path = output / "pilot-runtime.json"
    if state_path.exists():
        raise FileExistsError("Use a distinct pilot plan/output for another run")
    save(state_path, runtime)
    env = {
        **os.environ,
        "PYTHONPATH": str(source / "src"),
        "MUJOCO_GL": "egl",
        "TRANSFER_REPORT_ROOT": str(root),
    }

    def run(command):
        step = dict(command=command, started_utc=datetime.now(timezone.utc).isoformat())
        runtime["steps"].append(step)
        save(state_path, runtime)
        result = subprocess.run(command, cwd=source, env=env)
        step.update(returncode=result.returncode, ended_utc=datetime.now(timezone.utc).isoformat())
        save(state_path, runtime)
        result.check_returncode()

    try:
        run(plan["command"])
        dirs = list(Path(plan["log_root"]).glob("*transfer_v2_pilot*"))
        assert len(dirs) == 1, dirs
        run_dir = dirs[0]
        checkpoint = run_dir / "model_500.pt"
        history = (run_dir / "iterations.jsonl").read_text().splitlines()
        assert len(history) == 501
        runtime.update(
            run_directory=str(run_dir),
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint),
            updates=501,
            transitions=3078144,
        )
        save(state_path, runtime)
        for mode in ("clean", "dynamic"):
            evaluation = output / f"pilot-evaluation-{mode}.json"
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
                "20000",
                "--split",
                "validation",
                "--occlusion",
                mode,
                "--audit-termination",
                "--capture-representatives",
                str(output / f"pilot-traces-{mode}"),
                "--output",
                str(evaluation),
            ]
            run(command)
            report = json.loads(evaluation.read_text())
            assert report["episodes"] == 512
            assert set(report["per_variant"]) == {"nominal"}
            audit = report["termination_audit"]
            assert audit["success_hold_violations"] == audit["success_transfer_violations"] == 0
            assert digest(checkpoint) == runtime["checkpoint_sha256"]
            failures = [row for row in audit["records"] if not row["success"]]
            breakdown = {
                name: 0
                for name in (
                    "no_lift",
                    "not_placed",
                    "not_released",
                    "not_settled",
                    "not_withdrawn_or_held",
                )
            }
            for row in failures:
                key = next(
                    (
                        key
                        for key, flag in (
                            ("no_lift", "transfer_lifted"),
                            ("not_placed", "transfer_placed"),
                            ("not_released", "transfer_released"),
                            ("not_settled", "transfer_settled"),
                        )
                        if not row[flag]
                    ),
                    "not_withdrawn_or_held",
                )
                breakdown[key] += 1
            runtime.setdefault("evaluations", {})[mode] = dict(
                path=str(evaluation),
                successes=report["successes"],
                episodes=512,
                success_rate=report["success_rate"],
                failure_breakdown=breakdown,
                sha256=digest(evaluation),
            )
            save(state_path, runtime)
            run(
                [
                    plan["python"],
                    str(source / "render_transfer_capture.py"),
                    str(evaluation),
                    str(output / f"pilot-videos-{mode}"),
                ]
            )
        runtime["status"] = "completed"
    except BaseException as error:
        runtime.update(status="failed", error=repr(error))
        raise
    finally:
        runtime["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save(state_path, runtime)
        subprocess.run(
            [plan["python"], str(source / "build_transfer_repair_report.py")],
            cwd=source,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
