#!/usr/bin/env python3
"""Prepared privileged-controller feasibility audit of the actual Warp benchmark.

Import the frozen11a591f source via PYTHONPATH. Native proxies only forward copied
Warp state; no native stepping, native control IK, renderer, learner or videos.
Standard Warp RGB remains enabled for initial sensor-hash comparison.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from active_perception_arms import mdp, native, plug
from active_perception_arms.config import Experiment, saved_experiment, static_candidates
from active_perception_arms.environment import make_env
from active_perception_arms.evaluation_audit import TerminationAudit, summarize_termination_audit
from active_perception_arms.native import NativeEnv, ScriptedPolicy
from active_perception_arms.warp_sanity import sync_native_state

SOURCE_REVISION = "11a591f7a2129484d555caf115fcdbf1a709a0c9"
SEEDS = tuple(range(10000, 10004))
NUM_ENVS = 128


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_sha(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def cpu_snapshot(env, origins):
    """Batch the read-only copies; sync_native_state still performs all mapping."""
    arms = env.action_manager.get_term("arms")
    term = SimpleNamespace(
        targets={name: value.detach().cpu() for name, value in arms.targets.items()},
        tcp_target=arms.tcp_target.detach().cpu(),
        gimbal_target=arms.gimbal_target.detach().cpu(),
    )
    data = SimpleNamespace(
        **{
            name: getattr(env.sim.data, name).detach().cpu()
            for name in ("qpos", "qvel", "mocap_pos", "mocap_quat")
        }
    )
    return SimpleNamespace(
        sim=SimpleNamespace(data=data, mj_model=env.sim.mj_model),
        scene=SimpleNamespace(env_origins=origins),
        action_manager=SimpleNamespace(get_term=lambda name: term),
    )


def check_topology(proxy, model):
    if (proxy.model.nq, proxy.model.nv) != (model.nq, model.nv):
        raise RuntimeError("Native proxy coordinate dimensions differ from Warp")
    for joint in range(proxy.model.njnt):
        name = mujoco.mj_id2name(proxy.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if not np.array_equal(proxy.model.joint(name).qposadr, model.joint(name).qposadr):
            raise RuntimeError(f"Native proxy joint ordering differs: {name}")


class PhysicalAudit(TerminationAudit):
    """Add read-only terminal positions before the existing automatic reset."""

    def compute(self):
        done = super().compute()
        tcp, _, _, goal = mdp.positions(self.env)
        address = self.object_qpos_address
        arms = self.env.action_manager.get_term("arms")
        self.snapshot.update(
            body_qpos_position_world_m=self.env.sim.data.qpos[:, address : address + 3].clone(),
            body_goal_world_m=goal.clone(),
            tcp_derived_world_m=tcp.clone(),
            tcp_command_local_m=arms.tcp_target.clone(),
            finite_qpos=torch.isfinite(self.env.sim.data.qpos).all(dim=-1).clone(),
            episode_step=self.env.episode_length_buf.clone(),
        )
        return done


def compare_hashes(reference, cfg, combined, physics, sensors):
    expected = saved_experiment(reference["experiment"])
    # Saved JSON contains lists, while fresh dataclass coordinates are tuples.
    if (
        json.loads(json.dumps(expected.to_dict())) != json.loads(json.dumps(cfg.to_dict()))
        or reference["seed"] != SEEDS[0]
    ):
        return {"status": "CONFIGURATION_DIFFERS", "exact_initial_pairing_claimed": False}
    result = {"status": "COMPARABLE_CONFIGURATION", "episodes": len(combined)}
    for name, actual, prior in (
        ("combined", combined, reference.get("initial_state_sha256")),
        ("physics", physics, reference.get("initial_physics_state_sha256")),
        *[
            (name, values, reference.get("initial_sensor_sha256", {}).get(name))
            for name, values in sensors.items()
        ],
    ):
        if prior is None or len(prior) != len(actual):
            result[name] = {"status": "REFERENCE_HASHES_UNAVAILABLE_OR_COUNT_DIFFERS"}
        else:
            mismatches = [i for i, (a, b) in enumerate(zip(actual, prior, strict=True)) if a != b]
            result[name] = {"mismatches": len(mismatches), "episodes": mismatches}
    result["exact_initial_pairing_claimed"] = all(
        result[name].get("mismatches") == 0 for name in ("combined", "physics", *sensors)
    )
    return result


def run(args, cfg, provenance):
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This prepared audit requires an allocated CUDA GPU")
    timer = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    env = make_env(cfg, args.device)
    audit = PhysicalAudit(env)
    records, initial_records, combined_hashes, physics_hashes = [], [], [], []
    sensor_hashes = {name: [] for name in cfg.sensors}
    variants = [plug.VARIANTS[i] for i in plug.variant_assignment(NUM_ENVS)]
    proxy_models = {}
    try:
        # Match the evaluator's first reset followed by each explicitly seeded batch.
        env.reset(seed=SEEDS[0])
        origins = env.scene.env_origins.detach().cpu()
        proxies = []
        for variant in variants:
            proxy = NativeEnv(replace(cfg, num_envs=1, plug_variant=variant, render_sensors=False))
            check_topology(proxy, env.sim.mj_model)
            proxies.append(proxy)
            if variant not in proxy_models:
                proxy_models[variant] = model_sha(proxy.model)
        base_model_hash = model_sha(env.sim.mj_model)
        with torch.inference_mode():
            for batch, seed in enumerate(SEEDS):
                observations, _ = env.reset(seed=seed)
                audit.reset_batch()
                combined_hashes.extend(audit.initial_state_hashes(observations, cfg.sensors))
                physics, sensors = audit.initial_hash_components(observations, cfg.sensors)
                physics_hashes.extend(physics)
                for name, hashes in sensors.items():
                    sensor_hashes[name].extend(hashes)
                snapshot = cpu_snapshot(env, origins)
                clocks = env.episode_length_buf.detach().cpu().tolist()
                policies = []
                for i, proxy in enumerate(proxies):
                    sync_native_state(proxy, snapshot, int(clocks[i]), index=i)
                    # Controller must see the actual Warp spawn, not NativeEnv.reset.
                    policies.append(ScriptedPolicy(proxy))
                    initial_records.append(
                        {
                            "episode": batch * NUM_ENVS + i,
                            "reset_seed": seed,
                            "world": i,
                            "variant": variants[i],
                            "spawn_body_local_m": proxy.data.qpos[
                                proxy.object_adr : proxy.object_adr + 3
                            ].tolist(),
                            "socket_goal_local_m": proxy.data.site("fixture/goal").xpos.tolist(),
                            "body_goal_local_m": proxy.goal.tolist(),
                        }
                    )
                finished = np.zeros(NUM_ENVS, dtype=bool)
                minimum = np.full(NUM_ENVS, np.inf)
                clip_counts = np.zeros(NUM_ENVS, dtype=int)
                component_counts = np.zeros(NUM_ENVS, dtype=int)
                finite_all = np.ones(NUM_ENVS, dtype=bool)
                batch_records = [None] * NUM_ENVS
                steps = math.ceil(cfg.episode_seconds / cfg.step_dt)
                for step in range(steps):
                    action = np.zeros((NUM_ENVS, cfg.action_dim), dtype=np.float64)
                    for i in np.flatnonzero(~finished):
                        # For live first episodes this is the real episode step clock.
                        sync_native_state(proxies[i], snapshot, int(clocks[i]), index=int(i))
                        action[i] = policies[i]()
                        clip_counts[i] += int(np.count_nonzero(np.abs(action[i]) > 1))
                        component_counts[i] += cfg.action_dim
                    observations, _, terminated, truncated, _ = env.step(
                        torch.as_tensor(action, device=args.device, dtype=torch.float32)
                    )
                    snapshot_fields = {
                        name: value.detach().cpu() for name, value in audit.snapshot.items()
                    }
                    error = snapshot_fields["qpos_position_error_m"].numpy()
                    live = ~finished
                    minimum[live] = np.minimum(minimum[live], error[live])
                    finite_all[live] &= snapshot_fields["finite_qpos"].numpy()[live]
                    done = (terminated | truncated).detach().cpu().numpy()
                    first = done & live
                    for i in np.flatnonzero(first):
                        row = {
                            "episode": batch * NUM_ENVS + int(i),
                            "reset_seed": seed,
                            "world": int(i),
                            "variant": variants[i],
                            **{name: value[i].tolist() for name, value in snapshot_fields.items()},
                            "elapsed_s": (step + 1) * cfg.step_dt,
                            "minimum_qpos_error_m": float(minimum[i]),
                            "finite_qpos_all_preterminal_samples": bool(finite_all[i]),
                            "controller_stage_at_terminal_action": policies[i].stage,
                            "last_normalized_action": action[i].tolist(),
                            "clipped_action_components": int(clip_counts[i]),
                            "action_component_samples": int(component_counts[i]),
                        }
                        row["body_qpos_position_local_m"] = (
                            np.asarray(row["body_qpos_position_world_m"]) - origins[i].numpy()
                        ).tolist()
                        row["body_goal_local_m"] = (
                            np.asarray(row["body_goal_world_m"]) - origins[i].numpy()
                        ).tolist()
                        batch_records[i] = row
                    finished |= done
                    if finished.all():
                        break
                    snapshot = cpu_snapshot(env, origins)
                    clocks = env.episode_length_buf.detach().cpu().tolist()
                if any(row is None for row in batch_records):
                    raise RuntimeError("First-terminal scoring missed a world at the real timeout")
                records.extend(batch_records)
                print(
                    "batch",
                    batch,
                    "seed",
                    seed,
                    "successes",
                    sum(row["success"] for row in batch_records),
                    "/",
                    NUM_ENVS,
                    flush=True,
                )
    finally:
        audit.close()
        env.close()
    successes = sum(row["success"] for row in records)
    reference = json.loads(args.reference_report.read_text()) if args.reference_report else None
    summary = summarize_termination_audit(records, "plug")
    summary.pop("records")
    result = {
        "status": "COMPLETED",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "audit_wall_seconds": time.perf_counter() - timer,
        **provenance,
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": platform.node(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device),
            "packages": {
                name: version(name)
                for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch")
            },
        },
        "experiment": cfg.to_dict(),
        "reset_seeds": list(SEEDS),
        "scoring": "Existing enabled terminations; current qpos<2mm for3 consecutive25Hz samples; only first done per world. Native.step never called.",
        "nominal_horizon_s": cfg.episode_seconds,
        "maximum_control_steps": math.ceil(cfg.episode_seconds / cfg.step_dt),
        "quantized_timeout_s": math.ceil(cfg.episode_seconds / cfg.step_dt) * cfg.step_dt,
        "controller": "Unchanged privileged ScriptedPolicy; initialized after actual Warp state synchronization; CPU native FK only, real Warp control IK/physics.",
        "renderer_note": "Normal Warp RGB computed, only initial sensor hashes retained; no native renderer, learner, image/video output or learned visual inference.",
        "warp_host_base_model_mjb_sha256": base_model_hash,
        "native_proxy_model_mjb_sha256": proxy_models,
        "model_hash_note": "Warp host base model hash does not encode every per-world variant patch; variant assignment and frozen scene source are separately recorded.",
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "per_variant": {
            variant: {
                "episodes": sum(r["variant"] == variant for r in records),
                "successes": sum(r["success"] for r in records if r["variant"] == variant),
                "success_rate": float(
                    np.mean([r["success"] for r in records if r["variant"] == variant])
                ),
            }
            for variant in plug.VARIANTS
        },
        "termination_audit": summary,
        "initial_state_sha256": combined_hashes,
        "initial_physics_state_sha256": physics_hashes,
        "initial_sensor_sha256": sensor_hashes,
        "reference_report": str(args.reference_report) if args.reference_report else None,
        "reference_report_sha256": sha(args.reference_report) if args.reference_report else None,
        "reference_initial_comparison": compare_hashes(
            reference, cfg, combined_hashes, physics_hashes, sensor_hashes
        )
        if reference
        else {"status": "NO_REFERENCE"},
        "initial_records": initial_records,
        "records": records,
        "limits": [
            "This is a privileged controller/control-feasibility diagnostic, not visual learnability or perception utility.",
            "Success demonstrates feasibility for sampled Warp worlds; controller failure does not prove physical impossibility.",
            "An initial-hash match establishes only the initial pairing, not equivalence of controller trajectories or sources.",
            "Finished worlds may autoreset and receive zero actions, but only their first termination is scored.",
            "Native floating-point FK may differ from Warp derived poses; all physical success scores are the existing Warp manager's raw-qpos held-three criterion.",
            "Pre-reset TCP positions in records are explicitly derived and may lag by one physics substep; body qpos is current.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(json_safe(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": len(records),
                "successes": successes,
                "per_variant": result["per_variant"],
                "wall_seconds": result["audit_wall_seconds"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/plug_analysis/randomized_feasibility_gpu.json"),
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=Path("artifacts/plug_analysis/corrected-wrist_static-s0.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; preserve it and select a new output path")
    source = Path(native.__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        revision != SOURCE_REVISION
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    ):
        raise ValueError("Import clean frozen11a591f source via PYTHONPATH")
    cfg = Experiment(
        task="plug",
        condition="wrist_static",
        occlusion="clean",
        seed=SEEDS[0],
        num_envs=NUM_ENVS,
        randomize=True,
        fixed_position=static_candidates("plug")[7],
        success_state_sample="current_qpos",
    )
    modules = [
        "config.py",
        "native.py",
        "plug.py",
        "scenes.py",
        "environment.py",
        "mdp.py",
        "evaluation_audit.py",
        "warp_sanity.py",
        "robots/plug_control.py",
        "robots/kinematics.py",
    ]
    provenance = {
        "source_root": str(source),
        "source_revision": revision,
        "script_sha256": sha(__file__),
        "source_files_sha256": {
            name: sha(source / "src/active_perception_arms" / name) for name in modules
        },
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "PREPARED_NOT_SUBMITTED",
                    "experiment": cfg.to_dict(),
                    "episodes": NUM_ENVS * len(SEEDS),
                    "reset_seeds": list(SEEDS),
                    "termination_override": False,
                    "native_physics_steps": 0,
                    "reference_report": str(args.reference_report),
                    **provenance,
                },
                indent=2,
            )
        )
        return
    torch.set_num_threads(1)
    run(args, cfg, provenance)


if __name__ == "__main__":
    main()
