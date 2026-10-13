"""Frozen, bounded delta/GRU learning study with validation-gated plug curriculum."""

import argparse
import fcntl
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def updates(budget, num_envs):
    return math.ceil(budget / (num_envs * 24))


def stages():
    return [
        dict(name="ym_small_clean", variant="ym", jitter=0.002, horizon=8.0, occlusion="clean"),
        dict(name="all_small_clean", variant="all", jitter=0.002, horizon=8.0, occlusion="clean"),
        dict(name="all_full_clean", variant="all", jitter=0.008, horizon=8.0, occlusion="clean"),
        dict(
            name="all_full_dynamic", variant="all", jitter=0.008, horizon=8.0, occlusion="dynamic"
        ),
        dict(
            name="original_dynamic", variant="all", jitter=0.008, horizon=3.5, occlusion="dynamic"
        ),
    ]


def training_command(plan, case, output, budget, stage=None):
    task = case["task"]
    cmd = [
        plan["python"],
        "-m",
        "active_perception_arms.train",
        "--task",
        task,
        "--condition",
        case["condition"],
        "--control-revision",
        "pose_v1",
        "--translation-control",
        "delta",
        "--translation-step",
        "0.005",
        "--camera-workspace",
        "inspection_v2",
        "--camera-start",
        "inspection_v1",
        "--camera-interval",
        "4",
        "--memory",
        "gru",
        "--policy-profile",
        "compact",
        "--actor-observation",
        "privileged" if case["kind"] == "diagnostic" else "rgb",
        "--action-distribution",
        "squashed",
        "--initial-std",
        "0.3",
        "--entropy-coef",
        "0.003",
        "--learning-rate",
        "0.0003",
        "--lr-schedule",
        "adaptive",
        "--num-envs",
        str(plan["num_envs"]),
        "--seed",
        str(case["seed"]),
        "--width",
        "128",
        "--height",
        "96",
        "--initial-seconds",
        "1",
        "--iterations",
        str(updates(budget, plan["num_envs"])),
        "--save-interval",
        "100",
        "--job-type",
        "study",
        "--log-root",
        str(output / "logs"),
        "--run-label",
        case["name"] + "_" + output.name,
        "--result",
        str(output / f"training-{budget}.json"),
    ]
    if task == "plug":
        stage = stage or dict(variant="all", jitter=0.008, horizon=3.5, occlusion="dynamic")
        cmd += [
            "--reward-profile",
            "plug_insertion",
            "--plug-variant",
            stage["variant"],
            "--reset-xy-jitter",
            str(stage["jitter"]),
            "--episode-seconds",
            str(stage["horizon"]),
            "--occlusion",
            stage["occlusion"],
        ]
    else:
        cmd += [
            "--task-revision",
            "push_v2",
            "--reward-profile",
            "push_dense",
            "--push-workspace-revision",
            "arc_clear_v2",
            "--push-target-layout",
            "arc_v2",
            "--push-visibility-revision",
            "crossbar_v3",
            "--reset-xy-jitter",
            "0.02",
            "--episode-seconds",
            "12",
            "--occlusion",
            "dynamic",
        ]
    return cmd


def check_pairs(cases):
    for case in cases:
        if case["condition"] != "active":
            continue
        fixed = next(
            c
            for c in cases
            if c["kind"] == case["kind"]
            and c["task"] == case["task"]
            and c["seed"] == case["seed"]
            and c["condition"] == "wrist_static"
        )
        a, b = fixed["prepared"]["experiment"], case["prepared"]["experiment"]
        differences = {k for k in a if a[k] != b[k]}
        if differences != {"condition"}:
            raise ValueError(f"Unmatched fixed/active experiment: {differences}")
        for field in ("optimization", "initial_std", "action_distribution"):
            if fixed["prepared"].get(field) != case["prepared"].get(field):
                raise ValueError(f"Unmatched optimizer: {field}")


def prepare(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use the existing frozen plan or a new destination")
    source = output / "source"
    shutil.copytree(
        root / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in (Path(__file__).name, "validate_common_control_gpu.py"):
        shutil.copy2(root / "scripts" / name, source / name)
    shutil.copy2(root / "scripts/slurm/long_delta_study.sbatch", source / "long_delta_study.sbatch")
    plan = dict(
        root=str(root),
        output=str(output),
        source=str(source),
        python=str(root / ".venv/bin/python"),
        created_utc=datetime.now(timezone.utc).isoformat(),
        num_envs=256,
        milestones=[5_000_000, 10_000_000, 20_000_000],
        extension_milestone=50_000_000,
        extension_authorized=True,
        extension_rule="Extend both camera conditions for a task only if pooled three-seed mean validation success improves by >=2 percentage points, or mean terminal error drops >=10%, from 10M to 20M. Validation only; no 50M allocation before review.",
        validation_seed=130000,
        test_seed=140000,
        validation_episodes=512,
        test_episodes=1024,
        max_concurrent_gpus=8,
        curriculum_stages=stages(),
        curriculum_step=2_000_000,
        curriculum_stage_cap=4_000_000,
        curriculum_threshold=0.8,
        curriculum_transfer="Actor/critic weights and normalization transfer; fresh optimizer/std at each stage, exact resume within a stage.",
        selection="Maximum mean clean/dynamic validation success; earliest milestone breaks ties. Tests never select models or trigger promotion.",
        cases=[],
    )
    for task in ("plug", "push"):
        for seed in (0, 1, 2):
            for condition in ("wrist_static", "active"):
                plan["cases"].append(
                    dict(
                        name=f"{task}_{condition}_s{seed}",
                        task=task,
                        seed=seed,
                        condition=condition,
                        kind="direct",
                    )
                )
        plan["cases"].append(
            dict(
                name=f"{task}_privileged_s0",
                task=task,
                seed=0,
                condition="wrist_static",
                kind="diagnostic",
            )
        )
    for seed in (0, 1, 2):
        for condition in ("wrist_static", "active"):
            plan["cases"].append(
                dict(
                    name=f"plug_curriculum_{condition}_s{seed}",
                    task="plug",
                    seed=seed,
                    condition=condition,
                    kind="curriculum",
                )
            )
    env = os.environ | {"PYTHONPATH": str(source / "src")}
    for case in plan["cases"]:
        stage = stages()[0] if case["kind"] == "curriculum" else None
        command = training_command(plan, case, output / case["name"], 5_000_000, stage)
        case["prepared"] = json.loads(
            subprocess.check_output(command + ["--dry-run"], env=env, text=True)
        )
    check_pairs(plan["cases"])
    plan["source_sha256"] = {
        str(p.relative_to(source)): digest(p)
        for p in sorted(source.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    if getattr(args, "reuse_rendered_from", None):
        prior = args.reuse_rendered_from.resolve()
        old_plan = json.loads((prior / "plan.json").read_text())
        changed = {
            k for k, v in plan["source_sha256"].items() if old_plan["source_sha256"].get(k) != v
        }
        allowed = {
            "run_long_delta_study.py",
            "src/active_perception_arms/environment.py",
            "src/active_perception_arms/mdp.py",
            "src/active_perception_arms/evaluation_audit.py",
        }
        if not changed <= allowed:
            raise ValueError(
                f"Rendered gate reuse invalid after unrelated source changes: {changed}"
            )
        for name, expected in old_plan["source_sha256"].items():
            if digest(prior / "source" / name) != expected:
                raise ValueError("Prior frozen source was modified")
        report = prior / "gpu_gate/rendered.json"
        data = json.loads(report.read_text())
        if not data["passed"] or len(data["results"]) != 6:
            raise ValueError("Prior rendered gate did not pass all six cases")
        plan["rendered_gate_reuse"] = dict(
            path=str(report),
            sha256=digest(report),
            prior_plan=str(prior / "plan.json"),
            changed_sources=sorted(changed),
            reason="Only plug privileged metric routing, optional audit metrics and launcher changed; RGB controls, policy, reward and rendering remain unchanged.",
        )
    save(output / "plan.json", plan)
    summary(plan)
    print(json.dumps(dict(plan=str(output / "plan.json"), cases=len(plan["cases"]))))


def verify(plan):
    for name, expected in plan["source_sha256"].items():
        if digest(Path(plan["source"]) / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def physical_metrics(report):
    audit = report["termination_audit"]
    for key, value in audit.items():
        if key.endswith("violations") and value not in (None, 0):
            raise ValueError(f"Physical success audit failed: {key}={value}")
    import numpy as np

    records = audit["records"]
    errors = [r.get("push_xy_error", r["qpos_position_error_m"]) * 1000 for r in records]
    misses = [e for e, r in zip(errors, records, strict=True) if not r["success"]]
    result = dict(
        successes=report["successes"],
        episodes=report["episodes"],
        rate=report["success_rate"],
        mean_error_mm=float(np.mean(errors)),
        median_failed_error_mm=float(np.median(misses)) if misses else None,
        per_variant=report.get("per_variant"),
        physical_audit_passed=True,
    )
    for key in (
        "success_hold_steps",
        "push_withdrawn",
        "push_contact_free",
        "push_xy_error",
        "plug_xy_error_m",
        "plug_z_error_m",
        "plug_tcp_goal_clearance_m",
    ):
        values = [r[key] for r in records if key in r]
        if values:
            result[key + "_mean"] = float(np.mean(values))
    result["action_diagnostics"] = report.get("action_diagnostics")
    return result


def summary(plan):
    output = Path(plan["output"])
    with (output / "report.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for case in plan["cases"]:
            path = output / case["name"] / "state.json"
            state = json.loads(path.read_text()) if path.exists() else dict(status="pending")
            rows.append(
                dict(
                    name=case["name"],
                    task=case["task"],
                    kind=case["kind"],
                    condition=case["condition"],
                    seed=case["seed"],
                    **state,
                )
            )
        save(
            output / "summary.json",
            dict(created_utc=datetime.now(timezone.utc).isoformat(), rows=rows),
        )
        body = ""
        for row in rows:
            evaluations = row.get("evaluations", [])
            original = [
                e for e in evaluations if e["scope"] == "original" and e["split"] == "validation"
            ]
            latest = original[-1] if original else None
            score = f"{latest['rate'] * 100:.2f}% ({latest['occlusion']})" if latest else "Pending"
            body += (
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(x))}</td>"
                    for x in (
                        row["name"],
                        row["kind"],
                        row["status"],
                        row.get("transitions", 0),
                        score,
                    )
                )
                + "</tr>"
            )
        plots = ""
        for task in ("plug", "push"):
            lines = ""
            colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
            selected = [r for r in rows if r["task"] == task and r["kind"] == "direct"]
            xmax = max(
                [20_010_000]
                + [e["transitions"] for row in selected for e in row.get("evaluations", [])]
            )
            legend = ""
            for color, row in zip(colors, selected, strict=False):
                legend += f'<span style="color:{color};margin-right:16px">{html.escape(row["name"])}</span>'
                values = [
                    e
                    for e in row.get("evaluations", [])
                    if e["scope"] == "original"
                    and e["split"] == "validation"
                    and e["occlusion"] == "dynamic"
                ]
                points = " ".join(
                    f"{60 + 620 * e['transitions'] / xmax:.1f},{240 - 200 * e['rate']:.1f}"
                    for e in values
                )
                lines += (
                    f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'
                )
                for e in values:
                    x, y = 60 + 620 * e["transitions"] / xmax, 240 - 200 * e["rate"]
                    lines += f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"><title>{html.escape(row["name"])}: {e["rate"] * 100:.2f}% at {e["transitions"]:,}</title></circle>'
            plots += f'<h2>{task}: original-task dynamic validation</h2><svg viewBox="0 0 740 280" role="img" aria-label="{task} learning curves"><path d="M60 40V240H680" fill="none" stroke="black"/><text x="5" y="45">100%</text><text x="20" y="240">0%</text><text x="60" y="265">0</text><text x="595" y="265">{xmax / 1e6:.0f}M steps</text>{lines}</svg><p>{legend}</p>'
        document = f"""<!doctype html><html><meta charset="utf-8"><title>Long delta + GRU study</title>
        <style>body{{font:16px system-ui;margin:32px;background:white;color:#111;max-width:1100px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}svg{{width:100%;max-width:740px}}table{{border-collapse:collapse}}a{{color:#2563eb}}</style>
        <h1>Long delta-control + GRU study</h1><p>Fixed/active × three seeds for plug and corrected arc pushing. 5M/10M/20M milestones; 50M requires a progress review. Manipulator deltas: 5 mm; camera deltas: 10 mm / 0.05 rad, updated every four steps.</p>
        <p>Privileged diagnostics are state-informed actors, not visual results. Curriculum advances at 80% stage validation, with a 4M-step cap per stage. All original-task scores retain four plug variants, 3.5s horizon and 8mm jitter. Models are selected only by validation; final tests use a separate seed.</p>
        <p><a href="plan.json">Frozen plan</a> · <a href="summary.json">Live results and diagnostics</a> · <a href="../training_results/index.html">Previous training results</a></p>
        <table><tr><th>Policy</th><th>Branch</th><th>Status</th><th>Transitions</th><th>Latest original validation</th></tr>{body}</table>{plots}
        <p>Run the frozen worker's <code>report --plan plan.json</code> command to refresh. No new results are implied by pending rows.</p></html>"""
        temporary = output / "index.tmp"
        temporary.write_text(document)
        temporary.replace(output / "index.html")


def worker(args):
    plan = json.loads(args.plan.read_text())
    verify(plan)
    case = plan["cases"][args.index]
    extending = getattr(args, "extend", False)
    if extending:
        decision = json.loads((Path(plan["output"]) / "extension_review.json").read_text())
        if args.index not in decision["eligible_indices"]:
            raise ValueError("This policy did not meet the predeclared extension rule")
    output = Path(plan["output"]) / case["name"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "state.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else dict(evaluations=[], transitions=0)
        )
        if state.get("status") in ("curriculum_blocked", "skipped") or (
            state.get("status") == "complete" and (not extending or state.get("extension_complete"))
        ):
            return
        state.update(status="starting", job_id=os.environ.get("SLURM_JOB_ID"))
        env = os.environ | {
            "PYTHONPATH": str(Path(plan["source"]) / "src"),
            "MUJOCO_GL": "egl",
            "WANDB_RUN_GROUP": "long-delta-20261013",
            "WANDB_TAGS": case["kind"],
        }

        def persist():
            save(state_path, state)
            summary(plan)

        def call(cmd, label):
            if (Path(plan["output"]) / "suspension.json").exists():
                raise RuntimeError("Study suspended; no further training or evaluation")
            save(output / f"command-{label}.json", cmd)
            with (output / f"{label}.log").open("a") as log:
                subprocess.run(
                    cmd,
                    cwd=plan["source"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

        def evaluate(checkpoint, label, scope, split, occlusion, transitions, extra=()):
            path = output / f"{label}.json"
            cmd = [
                plan["python"],
                "-m",
                "active_perception_arms.evaluate",
                str(checkpoint),
                "--episodes",
                str(plan[split + "_episodes"]),
                "--num-envs",
                "128",
                "--seed",
                str(plan[split + "_seed"]),
                "--split",
                split,
                "--occlusion",
                occlusion,
                "--audit-termination",
                "--action-diagnostics",
                "--output",
                str(path),
                *extra,
            ]
            if scope == "original" and case["task"] == "plug":
                cmd += [
                    "--plug-variant",
                    "all",
                    "--episode-seconds",
                    "3.5",
                    "--reset-xy-jitter",
                    "0.008",
                ]
            if not path.exists():
                call(cmd, label)
            report = json.loads(path.read_text())
            result = dict(
                path=str(path),
                sha256=digest(path),
                checkpoint=str(checkpoint),
                checkpoint_sha256=digest(checkpoint),
                scope=scope,
                split=split,
                occlusion=occlusion,
                transitions=transitions,
                **physical_metrics(report),
            )
            state["evaluations"] = [e for e in state["evaluations"] if e["path"] != str(path)] + [
                result
            ]
            persist()
            return result

        def train(directory, budget, stage=None, initialize=None):
            result = directory / f"training-{budget}.json"
            u = updates(budget, plan["num_envs"])
            checkpoints = sorted(
                directory.glob("logs/*/model_*.pt"), key=lambda p: int(p.stem.split("_")[-1])
            )
            checkpoint = checkpoints[-1] if checkpoints else None
            if not result.exists():
                command = training_command(plan, case, directory, budget, stage)
                if checkpoint:
                    command += ["--resume", str(checkpoint)]
                elif initialize:
                    command += ["--initialize-from", str(initialize)]
                    if case["kind"] == "curriculum":
                        command += ["--initialize-actor-only"]
                state["status"] = "training"
                persist()
                call(command, directory.name + f"-train-{budget}")
            checkpoints = list(directory.glob(f"logs/*/model_{u - 1}.pt"))
            if len(checkpoints) != 1:
                raise RuntimeError(
                    f"Expected one completed checkpoint for {directory} at {u} updates"
                )
            return checkpoints[0], u * plan["num_envs"] * 24

        try:
            persist()
            if case["kind"] == "curriculum":
                # Launch only after the diagnostic and direct study finish; same seed/condition control.
                baseline = json.loads(
                    (
                        Path(plan["output"])
                        / f"plug_{case['condition']}_s{case['seed']}"
                        / "state-20M.json"
                    ).read_text()
                )
                diagnostic = json.loads(
                    (Path(plan["output"]) / "plug_privileged_s0" / "state-20M.json").read_text()
                )
                if baseline["status"] != "complete" or diagnostic["status"] != "complete":
                    raise RuntimeError("Curriculum prerequisites are incomplete")
                if (
                    max(e["rate"] for e in baseline["evaluations"] if e["split"] == "validation")
                    >= 0.8
                ):
                    state.update(
                        status="skipped",
                        reason="Direct original-task control already reached 80% validation",
                    )
                    persist()
                    return
                state["privileged_control_competent"] = (
                    max(e["rate"] for e in diagnostic["evaluations"] if e["split"] == "validation")
                    >= 0.8
                )
                previous, offset = None, 0
                for stage in plan["curriculum_stages"]:
                    directory = output / stage["name"]
                    passed = False
                    for budget in (plan["curriculum_step"], plan["curriculum_stage_cap"]):
                        checkpoint, steps = train(directory, budget, stage, previous)
                        state["transitions"] = offset + steps
                        stage_score = evaluate(
                            checkpoint,
                            f"{stage['name']}-{budget}-stage-validation",
                            "curriculum",
                            "validation",
                            stage["occlusion"],
                            offset + steps,
                        )
                        for mode in ("clean", "dynamic"):
                            evaluate(
                                checkpoint,
                                f"{stage['name']}-{budget}-original-{mode}",
                                "original",
                                "validation",
                                mode,
                                offset + steps,
                            )
                        if stage_score["rate"] >= plan["curriculum_threshold"]:
                            previous = checkpoint
                            passed = True
                            break
                    offset += steps
                    state["transitions"] = offset
                    state["curriculum_stage"] = stage["name"]
                    persist()
                    if not passed:
                        state["curriculum_stopped_at"] = stage["name"]
                        break
            else:
                for budget in [plan["extension_milestone"]] if extending else plan["milestones"]:
                    checkpoint, steps = train(output / "direct", budget)
                    state["transitions"] = steps
                    for mode in ("clean", "dynamic"):
                        evaluate(
                            checkpoint,
                            f"validation-{budget}-{mode}",
                            "original",
                            "validation",
                            mode,
                            steps,
                        )
            # Select only on original-task validation, including failed curriculum attempts.
            candidates = {}
            for e in state["evaluations"]:
                if e["scope"] == "original" and e["split"] == "validation":
                    candidates.setdefault(e["checkpoint"], []).append(e)
            ranked = [
                (sum(e["rate"] for e in es) / len(es), -es[0]["transitions"], ck)
                for ck, es in candidates.items()
                if {e["occlusion"] for e in es} == {"clean", "dynamic"}
            ]
            selected = Path(max(ranked)[2])
            selected_steps = candidates[str(selected)][0]["transitions"]
            save(
                output / ("selection-50M.json" if extending else "selection.json"),
                dict(checkpoint=str(selected), sha256=digest(selected), rule=plan["selection"]),
            )
            for mode in ("clean", "dynamic"):
                evaluate(
                    selected,
                    f"test-{'50M-' if extending else ''}{mode}",
                    "original",
                    "test",
                    mode,
                    selected_steps,
                )
            if case["condition"] == "active":
                for label, flags in (
                    ("held_image", ["--hold-external-after", "0"]),
                    ("frozen_camera", ["--freeze-camera-after", "1"]),
                    ("reset_memory", ["--reset-memory"]),
                ):
                    evaluate(
                        selected,
                        "intervention-" + ("50M-" if extending else "") + label,
                        "intervention",
                        "test",
                        "dynamic",
                        selected_steps,
                        flags,
                    )
            state["status"] = (
                "curriculum_blocked" if state.get("curriculum_stopped_at") else "complete"
            )
            state["selected_checkpoint"] = str(selected)
            if extending:
                state["extension_complete"] = True
            state["finished_utc"] = datetime.now(timezone.utc).isoformat()
            if case["kind"] != "curriculum" and not extending:
                save(output / "state-20M.json", state)
            persist()
        except BaseException as exc:
            state.update(status="failed", error=repr(exc))
            persist()
            raise


def gate(args):
    plan = json.loads(args.plan.read_text())
    verify(plan)
    env = os.environ | {"PYTHONPATH": str(Path(plan["source"]) / "src"), "MUJOCO_GL": "egl"}
    output = Path(plan["output"]) / plan.get("gate_directory", "gpu_gate")
    output.mkdir(exist_ok=True)
    if plan.get("rendered_gate_reuse"):
        prior = plan["rendered_gate_reuse"]
        if digest(Path(prior["path"])) != prior["sha256"]:
            raise ValueError("Prior rendered gate evidence changed")
        shutil.copy2(prior["path"], output / "rendered.json")
        save(output / "rendered_reuse.json", prior)
    else:
        subprocess.run(
            [
                plan["python"],
                str(Path(plan["source"]) / "validate_common_control_gpu.py"),
                "--inspection",
                "--push-v2",
                "--output",
                str(output / "rendered.json"),
            ],
            env=env,
            cwd=plan["source"],
            check=True,
        )
    tiny = plan | {"num_envs": 8}
    for task in () if plan.get("curriculum_gate") else ("plug", "push"):
        case = next(c for c in plan["cases"] if c["task"] == task and c["kind"] == "diagnostic")
        directory = output / (task + "_privileged")
        command = training_command(tiny, case, directory, 384)
        subprocess.run(command, env=env, cwd=plan["source"], check=True)
        checkpoint = next(directory.glob("logs/*/model_1.pt"))
        subprocess.run(
            training_command(tiny, case, directory, 768) + ["--resume", str(checkpoint)],
            env=env,
            cwd=plan["source"],
            check=True,
        )
    case = next(c for c in plan["cases"] if c["kind"] == "curriculum")
    previous = (
        Path(plan["curriculum_gate_initialize"]) if plan.get("curriculum_gate_initialize") else None
    )
    if previous and digest(previous) != plan["curriculum_gate_initialize_sha256"]:
        raise ValueError("Curriculum gate initialization checkpoint changed")
    for stage in stages()[2:] if plan.get("curriculum_gate") else stages()[:2]:
        directory = output / stage["name"]
        command = training_command(tiny, case, directory, 384, stage)
        if previous:
            command += ["--initialize-from", str(previous)]
            if plan.get("curriculum_gate"):
                command += ["--initialize-actor-only"]
        subprocess.run(command, env=env, cwd=plan["source"], check=True)
        previous = next(directory.glob("logs/*/model_1.pt"))
        subprocess.run(
            [
                plan["python"],
                "-m",
                "active_perception_arms.evaluate",
                str(previous),
                "--episodes",
                "8",
                "--num-envs",
                "8",
                "--seed",
                "150000",
                "--plug-variant",
                "all",
                "--episode-seconds",
                "3.5",
                "--reset-xy-jitter",
                "0.008",
                "--occlusion",
                "dynamic",
                "--audit-termination",
                "--output",
                str(directory / "original-evaluation.json"),
            ],
            env=env,
            cwd=plan["source"],
            check=True,
        )
        physical_metrics(json.loads((directory / "original-evaluation.json").read_text()))
    save(
        output / "passed.json",
        dict(
            passed=True,
            source_sha256=plan["source_sha256"],
            checks=(
                [
                    "actor-only transfer across jitter, occlusion and horizon stages",
                    "original-task scoring",
                ]
                if plan.get("curriculum_gate")
                else [
                    "rendered PPO",
                    "fixed/active image parity",
                    "privileged GRU",
                    "exact resume",
                    "curriculum weight transfer",
                    "original-task scoring",
                ]
            ),
        ),
    )


def submit(args):
    plan = json.loads(args.plan.read_text())
    verify(plan)
    output = Path(plan["output"])
    with (output / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "submission.json").exists():
            raise FileExistsError("Submission already recorded; do not duplicate jobs")
        if (output / "suspension.json").exists():
            raise RuntimeError("Study suspended")
        base = [
            "sbatch",
            "--parsable",
            "--account=kempner_pgozdil_lab",
            "--partition=kempner_rtx",
            "--exclude=holygpu7c2313",
            "--export=ALL",
        ]
        wrapper = str(Path(plan["source"]) / "long_delta_study.sbatch")
        record = dict(commands=[], jobs={})

        def launch(name, flags, action):
            command = (
                base
                + ["--job-name=long-delta-" + name]
                + flags
                + [wrapper, str(args.plan.resolve()), action]
            )
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            record["commands"].append(command)
            record["jobs"][name] = job
            save(output / "submission.json", record)
            return job

        gate_job = launch("gate", ["--time=01:00:00"], "gate")
        indices = [str(i) for i, c in enumerate(plan["cases"]) if c["kind"] != "curriculum"]
        main_job = launch(
            "main",
            [
                "--time=12:00:00",
                "--array=" + ",".join(indices) + "%8",
                "--dependency=afterok:" + gate_job,
            ],
            "run",
        )
        indices = [str(i) for i, c in enumerate(plan["cases"]) if c["kind"] == "curriculum"]
        launch(
            "curriculum",
            [
                "--time=12:00:00",
                "--array=" + ",".join(indices) + "%6",
                "--dependency=afterok:" + main_job,
            ],
            "run",
        )
        launch(
            "review",
            [
                "--time=00:15:00",
                "--gres=none",
                "--partition=shared",
                "--account=pgozdil_lab",
                "--cpus-per-task=2",
                "--mem=8G",
                "--dependency=afterok:" + main_job,
            ],
            "review",
        )
        print(json.dumps(record["jobs"]))


def extension_decision(plan):
    """Use every matched seed and condition; test results cannot enter the decision."""
    results, eligible = {}, []
    for task in ("plug", "push"):
        buckets = {10_000_000: [], 20_000_000: []}
        indices = [
            i for i, c in enumerate(plan["cases"]) if c["task"] == task and c["kind"] == "direct"
        ]
        for index in indices:
            case = plan["cases"][index]
            state = json.loads((Path(plan["output"]) / case["name"] / "state.json").read_text())
            if state["status"] != "complete":
                raise ValueError("Review requires every direct policy to finish")
            for budget in buckets:
                steps = updates(budget, plan["num_envs"]) * plan["num_envs"] * 24
                evaluations = [
                    e
                    for e in state["evaluations"]
                    if e["scope"] == "original"
                    and e["split"] == "validation"
                    and e["transitions"] == steps
                ]
                if {e["occlusion"] for e in evaluations} != {"clean", "dynamic"} or len(
                    evaluations
                ) != 2:
                    raise ValueError("Missing matched clean/dynamic milestone validations")
                buckets[budget].extend(evaluations)
        means = {
            b: dict(
                rate=sum(e["rate"] for e in es) / len(es),
                error_mm=sum(e["mean_error_mm"] for e in es) / len(es),
            )
            for b, es in buckets.items()
        }
        gain = means[20_000_000]["rate"] - means[10_000_000]["rate"]
        old_error = means[10_000_000]["error_mm"]
        reduction = 1 - means[20_000_000]["error_mm"] / old_error if old_error > 0 else 0
        extend = gain >= 0.02 or reduction >= 0.10
        results[task] = dict(
            means=means,
            success_gain_pp=100 * gain,
            error_reduction_fraction=reduction,
            extend=extend,
        )
        if extend:
            eligible.extend(indices)
    return dict(tasks=results, eligible_indices=eligible, rule=plan["extension_rule"])


def review(args):
    plan = json.loads(args.plan.read_text())
    verify(plan)
    output = Path(plan["output"])
    with (output / "extension.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "extension_review.json").exists():
            raise FileExistsError("Extension review already recorded; do not duplicate jobs")
        decision = extension_decision(plan)
        save(output / "extension_review.json", decision)
        if decision["eligible_indices"]:
            if (output / "suspension.json").exists():
                raise RuntimeError("Study suspended; extension not submitted")
            command = [
                "sbatch",
                "--parsable",
                "--account=kempner_pgozdil_lab",
                "--partition=kempner_rtx",
                "--exclude=holygpu7c2313",
                "--export=ALL",
                "--job-name=long-delta-50M",
                "--time=12:00:00",
                "--array=" + ",".join(map(str, decision["eligible_indices"])) + "%2",
                str(Path(plan["source"]) / "long_delta_study.sbatch"),
                str(args.plan.resolve()),
                "extend",
            ]
            decision["command"] = command
            decision["job_id"] = subprocess.check_output(command, text=True).strip().split(";")[0]
            save(output / "extension_review.json", decision)
        summary(plan)
        print(json.dumps(decision))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-rendered-from", type=Path)
    for action in ("run", "gate", "submit", "report", "review"):
        p = sub.add_parser(action)
        p.add_argument("--plan", type=Path, required=True)
        if action == "run":
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    if args.action == "report":
        summary(json.loads(args.plan.read_text()))
    else:
        {"prepare": prepare, "run": worker, "gate": gate, "submit": submit, "review": review}[
            args.action
        ](args)
