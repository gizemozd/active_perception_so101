"""Read-only reconstruction of restored-plug Slurm jobs and retained run artifacts."""

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--start", default="2026-10-07")
p.add_argument("--output", type=Path, default=Path("artifacts/cluster_pilot/run_inventory.json"))
a = p.parse_args()
fields = "JobID,JobIDRaw,JobName,State,Submit,Start,End,Elapsed,ExitCode,NodeList,Account,Partition,ReqTRES,AllocTRES,WorkDir,SubmitLine"
result = subprocess.run(
    ["sacct", "-u", "pgozdil", "-S", a.start, "-X", "-P", "--format=" + fields],
    check=True,
    capture_output=True,
    text=True,
)
lines = result.stdout.splitlines()
jobs = []
for line in lines[1:]:
    row = dict(zip(lines[0].split("|"), line.split("|"), strict=True))
    if row["JobName"] not in (
        "plug-inventory",
        "plug-colocation",
        "vision-validate",
        "plug-benchmark",
        "vision-plug",
        "vision-evaluate",
    ):
        continue
    row["scontrol"] = "Terminal state from sacct; controller not queried by this snapshot"
    if row["State"] in ("RUNNING", "PENDING", "COMPLETING"):
        row["scontrol"] = (
            subprocess.run(
                ["scontrol", "show", "job", row["JobID"]],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
            or "No controller output; accounting retained"
        )
    row["runs"] = []
    jobs.append(row)
for path in Path("logs").glob("**/runtime_*.json"):
    runtime = json.loads(path.read_text())
    directory = path.parent
    for job in jobs:
        # SLURM_JOB_ID can be the array task's underlying numeric allocation ID.
        ids = {job["JobIDRaw"]}
        control = job["scontrol"].split()
        ids.update(s.split("=")[1] for s in control if s.startswith("JobId="))
        if runtime.get("slurm_job_id") not in ids:
            continue
        experiment = json.loads((directory / "experiment.json").read_text())
        runner = json.loads((directory / "runner.json").read_text())
        history_path = directory / "iterations.jsonl"
        history = (
            [json.loads(line) for line in history_path.read_text().splitlines()]
            if history_path.exists()
            else []
        )
        wandb_path = directory / "wandb_run.json"
        summary_path = directory / "summary.json"
        job["runs"].append(
            dict(
                directory=str(directory),
                runtime=runtime,
                experiment=experiment,
                configured_iterations=runner["max_iterations"],
                configured_transitions=runner["max_iterations"]
                * runner["num_steps_per_env"]
                * experiment["num_envs"],
                latest_progress=history[-1] if history else None,
                checkpoint_paths=[str(x) for x in sorted(directory.glob("*.pt"))],
                wandb=json.loads(wandb_path.read_text()) if wandb_path.exists() else None,
                summary=str(summary_path) if summary_path.exists() else None,
            )
        )
manifest = a.output.parent / "pilot_manifest.json"
if manifest.exists():
    for pilot in json.loads(manifest.read_text()):
        for prefix in ("evaluation", "recorded_repeat"):
            for job in jobs:
                if job["JobIDRaw"] != pilot.get(prefix + "_job_id"):
                    continue
                path = Path(pilot[prefix + "_output"])
                report = json.loads(path.read_text()) if path.exists() else {}
                job["evaluation"] = dict(
                    report=str(path),
                    checkpoint=pilot["final_checkpoint"],
                    source_revision=pilot.get(prefix + "_source_revision"),
                    condition=pilot["condition"],
                    configured_episodes=512,
                    num_envs=128,
                    reset_seeds=[10000, 10001, 10002, 10003],
                    completed_episodes=report.get("episodes", 0),
                    successes=report.get("successes"),
                    wandb_url=report.get("wandb_url"),
                    media=str(path.with_name(path.stem + "-videos")),
                    diagnostic_repeat=prefix == "recorded_repeat",
                )
a.output.write_text(
    json.dumps(
        dict(
            checked_at=dt.datetime.now().astimezone().isoformat(),
            jobs=jobs,
            queue=subprocess.run(
                ["squeue", "-r", "-u", "pgozdil", "-p", "kempner_rtx", "-h", "-o", "%i|%T|%M|%R"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines(),
        ),
        indent=2,
    )
)
for job in jobs:
    print(
        job["JobID"],
        job["State"],
        job["Elapsed"],
        [
            (
                r["experiment"]["condition"],
                r["experiment"]["num_envs"],
                r["latest_progress"]["transitions"] if r["latest_progress"] else 0,
            )
            for r in job["runs"]
        ],
    )
