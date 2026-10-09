"""Checkpoint evaluation and causal interventions; never updates model parameters."""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from rsl_rl.utils import resolve_callable

from . import plug
from .config import OCCLUSIONS, SUCCESS_STATE_SAMPLES, saved_experiment
from .environment import make_env
from .occlusion import with_occlusion


def load_actor(checkpoint, observations, device):
    config = json.loads(checkpoint.with_name("runner.json").read_text())
    options = copy.deepcopy(config["actor"])
    cls = resolve_callable(options.pop("class_name"))
    options = {k: v for k, v in options.items() if v is not None}
    experiment = saved_experiment(json.loads(checkpoint.with_name("experiment.json").read_text()))
    actor = cls(observations, config["obs_groups"], "actor", experiment.action_dim, **options).to(
        device
    )
    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(weights["actor_state_dict"])
    return actor.eval()


def wilson(successes, total):
    if total < 1:
        raise ValueError("At least one evaluation episode is required")
    z = 1.95996398454
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def evaluate(args):
    start_time = time.perf_counter()
    if args.episodes < 1 or args.num_envs < 1:
        raise ValueError("episodes and num-envs must be positive")
    original = saved_experiment(
        json.loads(args.checkpoint.with_name("experiment.json").read_text())
    )
    overrides = {}
    if getattr(args, "success_state_sample", None) is not None:
        overrides["success_state_sample"] = args.success_state_sample
    cfg = with_occlusion(
        original,
        args.occlusion,
        num_envs=min(args.num_envs, args.episodes),
        seed=args.seed,
        **overrides,
    )
    if (
        args.freeze_camera_after is not None or args.camera_trace_in or args.camera_trace_out
    ) and not cfg.camera_control:
        raise ValueError("Camera interventions require an active or initial camera checkpoint")
    steps = math.ceil(cfg.episode_seconds / cfg.step_dt)
    batches = math.ceil(args.episodes / cfg.num_envs)
    replay = None
    if args.camera_trace_in:
        with np.load(args.camera_trace_in) as data:
            replay = torch.as_tensor(data["actions"], device=args.device)
            if str(data["task"]) != cfg.task or float(data["step_dt"]) != cfg.step_dt:
                raise ValueError("Replay task/control rate mismatch")
        if replay.shape != (batches, steps, cfg.num_envs, 5):
            raise ValueError("Replay must use identical episode count, num-envs, and horizon")
    traces = []
    outcomes, times, distances, variants = [], [], [], []
    capture = None
    capture_path = None
    if getattr(args, "capture_representatives", None):
        from .capture_evaluation import RepresentativeCapture

        capture = RepresentativeCapture(args.capture_representatives, cfg)
    env = make_env(cfg, args.device)
    audit = None
    audit_records, initial_hashes, initial_physics_hashes = [], [], []
    initial_sensor_hashes = {name: [] for name in cfg.sensors}
    if getattr(args, "audit_termination", False):
        from .evaluation_audit import TerminationAudit

        audit = TerminationAudit(env)
    try:
        obs, _ = env.reset(seed=args.seed)
        actor = load_actor(args.checkpoint, obs, args.device)
        with torch.inference_mode():
            for batch in range(batches):
                env.action_manager.get_term("arms").camera_frozen = False
                obs, _ = env.reset(seed=args.seed + batch)
                batch_audit = [None] * cfg.num_envs
                if audit:
                    audit.reset_batch()
                    initial_hashes.extend(audit.initial_state_hashes(obs, cfg.sensors))
                    physics_hashes, sensor_hashes = audit.initial_hash_components(obs, cfg.sensors)
                    initial_physics_hashes.extend(physics_hashes)
                    for name, values in sensor_hashes.items():
                        initial_sensor_hashes[name].extend(values)
                actor.reset()
                finished = torch.zeros(cfg.num_envs, device=args.device, dtype=torch.bool)
                won = torch.zeros_like(finished)
                elapsed = torch.full((cfg.num_envs,), cfg.episode_seconds, device=args.device)
                motion = torch.zeros(cfg.num_envs, device=args.device)
                frozen = None
                recorded = []
                for step in range(steps):
                    t = step * cfg.step_dt
                    if args.reset_memory:
                        actor.reset()
                    if args.hold_external_after is not None and t >= args.hold_external_after:
                        if "external" not in obs:
                            raise ValueError("External image hold requires an external camera")
                        if frozen is None:
                            frozen = obs["external"].clone()
                        obs["external"] = frozen
                    action = actor(obs).clone()
                    if cfg.camera_control:
                        if replay is not None:
                            action[:, cfg.manip_dim :] = replay[batch, step]
                        if args.freeze_camera_after is not None and t >= args.freeze_camera_after:
                            action[:, cfg.manip_dim :] = 0
                            # Zero gimbal deltas would still let iterative IK move.
                            env.action_manager.get_term("arms").camera_frozen = True
                        if args.camera_trace_out:
                            recorded.append(action[:, cfg.manip_dim :].cpu().numpy())
                    before = env.scene["camera_arm"].data.joint_pos.clone()
                    if capture:
                        capture.step(env, obs, action)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    if capture:
                        capture.reward(reward)
                    done = terminated | truncated
                    first = done & ~finished
                    if audit:
                        for i in first.nonzero(as_tuple=False).flatten().cpu().tolist():
                            batch_audit[i] = {
                                "episode": batch * cfg.num_envs + i,
                                **{
                                    name: value[i].cpu().tolist()
                                    for name, value in audit.snapshot.items()
                                },
                            }
                    # Termination manager retains this step's flags through automatic reset.
                    won |= first & env.termination_manager.get_term("success")
                    elapsed[first] = (step + 1) * cfg.step_dt
                    delta = (env.scene["camera_arm"].data.joint_pos - before).abs().sum(-1)
                    # Exclude terminal reset jumps from physical-motion diagnostics.
                    motion += delta * (~finished & ~done)
                    finished |= done
                    actor.reset(done)
                count = min(cfg.num_envs, args.episodes - len(outcomes))
                variants.extend(
                    [
                        plug.VARIANTS[i]
                        for i in plug.variant_assignment(cfg.num_envs, cfg.plug_variant)
                    ][:count]
                )
                outcomes.extend(won[:count].cpu().tolist())
                times.extend(elapsed[:count].cpu().tolist())
                distances.extend(motion[:count].cpu().tolist())
                if recorded:
                    traces.append(np.stack(recorded))
                if capture:
                    capture_path = capture.finish_batch(batch, won, elapsed, count)
                if audit:
                    if any(row is None for row in batch_audit[:count]):
                        raise RuntimeError("Termination audit missed an evaluated episode")
                    audit_records.extend(batch_audit[:count])
    finally:
        if audit:
            audit.close()
        env.close()
    if args.camera_trace_out:
        args.camera_trace_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.camera_trace_out,
            actions=np.stack(traces),
            task=cfg.task,
            step_dt=cfg.step_dt,
            source_seed=args.seed,
            source_checkpoint=str(args.checkpoint),
        )
    successes = sum(outcomes)
    report = {
        "checkpoint": str(args.checkpoint),
        "experiment": cfg.to_dict(),
        "split": args.split,
        "seed": args.seed,
        "training_seed": original.seed,
        "episodes": len(outcomes),
        "successes": successes,
        "success_rate": successes / len(outcomes),
        "wilson_95": wilson(successes, len(outcomes)),
        "variants": variants,
        "per_variant": {
            v: {
                "episodes": sum(x == v for x in variants),
                "successes": sum(w for w, x in zip(outcomes, variants, strict=True) if x == v),
                "success_rate": float(
                    np.mean([w for w, x in zip(outcomes, variants, strict=True) if x == v])
                ),
            }
            for v in sorted(set(variants))
        },
        "mean_episode_seconds": float(np.mean(times)),
        "mean_success_completion_seconds": float(
            np.mean([t for t, w in zip(times, outcomes, strict=True) if w])
        )
        if successes
        else None,
        "mean_camera_joint_travel_rad": float(np.mean(distances)),
        "evaluation_wall_seconds": time.perf_counter() - start_time,
        "outcomes": outcomes,
        "episode_seconds": times,
        "camera_joint_travel_rad": distances,
        "interventions": {
            "freeze_camera_after": args.freeze_camera_after,
            "hold_external_after": args.hold_external_after,
            "reset_memory": args.reset_memory,
            "camera_trace_in": str(args.camera_trace_in) if args.camera_trace_in else None,
            "success_state_sample_override": getattr(args, "success_state_sample", None),
        },
        "training_started": False,
        "representative_capture": capture_path,
        "reference_report": str(args.reference_report)
        if getattr(args, "reference_report", None)
        else None,
    }
    if audit:
        from .evaluation_audit import summarize_termination_audit

        report["termination_audit"] = summarize_termination_audit(audit_records, cfg.task)
        report["initial_state_sha256"] = initial_hashes[: len(outcomes)]
        report["initial_physics_state_sha256"] = initial_physics_hashes[: len(outcomes)]
        report["initial_sensor_sha256"] = {
            name: values[: len(outcomes)] for name, values in initial_sensor_hashes.items()
        }
    if getattr(args, "reference_report", None):
        reference = json.loads(args.reference_report.read_text())
        if len(reference["outcomes"]) != len(outcomes):
            raise ValueError("Reference report has a different episode count")
        changed = [
            i
            for i, (before, after) in enumerate(zip(reference["outcomes"], outcomes, strict=True))
            if before != after
        ]
        previous_hashes = reference.get("initial_state_sha256")
        previous_physics = reference.get("initial_physics_state_sha256")
        previous_sensors = reference.get("initial_sensor_sha256", {})
        report["repeat_comparison"] = {
            "reference_successes": reference["successes"],
            "success_count_change": successes - reference["successes"],
            "changed_episode_count": len(changed),
            "changed_episode_indices": changed,
            "initial_hash_comparable": bool(previous_hashes and audit),
            "initial_hash_mismatches": sum(
                a != b
                for a, b in zip(previous_hashes, initial_hashes[: len(outcomes)], strict=True)
            )
            if previous_hashes and audit
            else None,
            "initial_physics_state_mismatches": sum(
                a != b
                for a, b in zip(
                    previous_physics, initial_physics_hashes[: len(outcomes)], strict=True
                )
            )
            if previous_physics and audit
            else None,
            "initial_sensor_image_mismatches": {
                name: sum(
                    a != b
                    for a, b in zip(previous_sensors[name], values[: len(outcomes)], strict=True)
                )
                for name, values in initial_sensor_hashes.items()
                if name in previous_sensors
            }
            if audit
            else None,
            "same_success_state_sample": reference["experiment"].get(
                "success_state_sample", "derived_substep"
            )
            == cfg.success_state_sample,
            "note": "Repeated seeded GPU execution need not reproduce every trajectory; hashes compare state plus actor images when both reports contain them.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    compact = {
        k: v
        for k, v in report.items()
        if k
        not in (
            "outcomes",
            "episode_seconds",
            "camera_joint_travel_rad",
            "initial_state_sha256",
            "initial_physics_state_sha256",
            "initial_sensor_sha256",
        )
    }
    if "termination_audit" in compact:
        compact["termination_audit"] = {
            k: v for k, v in compact["termination_audit"].items() if k != "records"
        }
    return compact


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--occlusion", choices=OCCLUSIONS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--freeze-camera-after", type=float)
    p.add_argument("--hold-external-after", type=float)
    p.add_argument("--reset-memory", action="store_true")
    p.add_argument("--camera-trace-in", type=Path)
    p.add_argument("--camera-trace-out", type=Path)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--capture-representatives", type=Path)
    p.add_argument("--reference-report", type=Path)
    p.add_argument(
        "--success-state-sample",
        choices=SUCCESS_STATE_SAMPLES,
        help="Explicit criterion intervention; default retains the checkpoint's saved sampling",
    )
    p.add_argument(
        "--audit-termination",
        action="store_true",
        help="Record success-time error/hold before automatic resets; reads only",
    )
    p.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = p.parse_args(argv)
    args.seed = (
        args.seed if args.seed is not None else (10000 if args.split == "validation" else 20000)
    )
    torch.set_num_threads(4 if args.device.startswith("cuda") else 1)
    report = evaluate(args)
    if args.wandb:
        import wandb

        parent = json.loads(args.checkpoint.with_name("wandb_run.json").read_text())
        run = wandb.init(
            project=parent["project"],
            entity=parent["entity"],
            group=os.getenv("WANDB_RUN_GROUP", "plug-pilot-20261007"),
            job_type="validation-recorded-repeat" if args.reference_report else "validation",
            name=args.checkpoint.parent.name
            + ("_recorded_repeat" if args.reference_report else "_validation"),
            tags=[
                os.getenv("EVALUATION_STAGE", "pilot"),
                "validation",
                report["experiment"]["condition"],
            ],
            config={
                "evaluation": report,
                "training_run": parent,
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            },
        )
        checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
        original = saved_experiment(
            json.loads(args.checkpoint.with_name("experiment.json").read_text())
        )
        transitions = (checkpoint["iter"] + 1) * original.num_envs * 24
        run.define_metric("validation/*", step_metric="train/transitions")
        metrics = {
            "train/transitions": transitions,
            "validation/success": report["success_rate"],
            "validation/mean_episode_seconds": report["mean_episode_seconds"],
            "validation/mean_camera_joint_travel_rad": report["mean_camera_joint_travel_rad"],
        }
        if report["mean_success_completion_seconds"] is not None:
            metrics["validation/completion_seconds"] = report["mean_success_completion_seconds"]
        metrics.update(
            {"validation/success_" + v: r["success_rate"] for v, r in report["per_variant"].items()}
        )
        run.log(metrics)
        full = json.loads(args.output.read_text())
        full["wandb_url"] = run.url
        full["wandb_id"] = run.id
        args.output.write_text(json.dumps(full, indent=2))
        run.save(str(args.output), base_path=str(args.output.parent))
        run.finish()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
