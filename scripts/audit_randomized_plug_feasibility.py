#!/usr/bin/env python3
"""Sample randomized native feasibility with the unchanged privileged controller.

Run with PYTHONPATH pointing to the frozen plug-bootstrap-20261009/src checkout.
This does not render images, learn a policy, or establish Warp/visual feasibility.
"""

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import torch

from active_perception_arms import native, plug
from active_perception_arms.config import Experiment, static_candidates
from active_perception_arms.native import NativeEnv, ScriptedPolicy

SOURCE_REVISION = "11a591f7a2129484d555caf115fcdbf1a709a0c9"
SEEDS = tuple(range(10000, 10016))
HOLD_SAMPLES = 3


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        data = np.ascontiguousarray(array)
        digest.update(str(data.shape).encode())
        digest.update(str(data.dtype).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def model_sha(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def rollout(env, seed):
    env.reset(seed)
    policy = ScriptedPolicy(env)
    adr = env.object_adr
    initial_qpos = env.data.qpos.copy()
    initial_mocap = env.data.mocap_pos.copy()
    goal = env.goal.copy()  # World socket goal with the hidden prong offset removed.
    result = {
        "seed": seed,
        "variant": env.variant,
        "initial_state_sha256": array_sha(initial_qpos, env.data.qvel, initial_mocap),
        "spawn_body_position_m": initial_qpos[adr : adr + 3].tolist(),
        "spawn_tcp_target_m": env.tcp_target.tolist(),
        "socket_mocap_position_m": env.data.mocap_pos[env.fixture_id].tolist(),
        "socket_goal_world_position_m": env.data.site("fixture/goal").xpos.tolist(),
        "offset_corrected_body_goal_m": goal.tolist(),
        "initial_error_mm": float(np.linalg.norm(initial_qpos[adr : adr + 3] - goal) * 1000),
    }
    trace = []
    held = 0
    finite = True
    first_instant = None
    first_held = None
    native_disagreements = 0
    horizon_steps = math.ceil(env.cfg.episode_seconds / env.cfg.step_dt)
    for step in range(1, horizon_steps + 1):
        action = np.asarray(policy(), dtype=float)
        instantaneous_native = env.step(action)  # Instantaneous return is NOT the score.
        finite = finite and bool(np.isfinite(env.data.qpos).all())
        body = env.data.qpos[adr : adr + 3].copy()
        current_goal = env.goal.copy()
        error = float(np.linalg.norm(body - current_goal))
        inside = finite and error < plug.SUCCESS_DISTANCE
        held = held + 1 if inside else 0
        native_disagreements += int(bool(instantaneous_native) != inside)
        if inside and first_instant is None:
            first_instant = step
        trace.append(
            {
                "step": step,
                "time_s": float(env.data.time),
                "error_mm": error * 1000 if math.isfinite(error) else None,
                "hold_samples": held,
                "body_qpos_position_m": body.tolist() if np.isfinite(body).all() else None,
                "tcp_world_position_m": env.tcp.tolist(),
                "tcp_command_target_m": env.tcp_target.tolist(),
                "tcp_tracking_error_mm": float(np.linalg.norm(env.tcp - env.tcp_target) * 1000),
                "manipulator_joint_position": env.data.qpos[env.qadr["manipulator"]].tolist(),
                "manipulator_joint_target": env.targets["manipulator"].tolist(),
                "action_normalized_raw": action.tolist(),
                "action_normalized_clipped": np.clip(action, -1, 1).tolist(),
                "action_clip_components": int(np.count_nonzero(np.abs(action) > 1)),
                "finite_qpos": finite,
                "controller_stage": policy.stage,
            }
        )
        if held >= HOLD_SAMPLES:
            first_held = step
            break  # Same success termination: never score post-success behavior.
        if not finite:
            break
    errors = [row["error_mm"] for row in trace if row["error_mm"] is not None]
    actions = np.array([row["action_normalized_raw"] for row in trace])
    result.update(
        success=first_held is not None,
        earliest_instant_step=first_instant,
        earliest_instant_time_s=first_instant * env.cfg.step_dt if first_instant else None,
        earliest_held_step=first_held,
        earliest_held_time_s=first_held * env.cfg.step_dt if first_held else None,
        held_before_nominal_horizon=bool(
            first_held is not None
            and first_held * env.cfg.step_dt <= env.cfg.episode_seconds + 1e-12
        ),
        stopped_after_steps=len(trace),
        stopped_time_s=float(env.data.time),
        stop_reason="held_success" if first_held else ("timeout" if finite else "nonfinite_qpos"),
        minimum_error_mm=min(errors) if errors else None,
        terminal_error_mm=trace[-1]["error_mm"],
        terminal_hold_samples=held,
        terminal_body_qpos_position_m=trace[-1]["body_qpos_position_m"],
        terminal_tcp_tracking_error_mm=trace[-1]["tcp_tracking_error_mm"],
        terminal_controller_stage=trace[-1]["controller_stage"],
        hold_error_mm=[row["error_mm"] for row in trace[-HOLD_SAMPLES:]] if first_held else [],
        finite_qpos_all_samples=finite,
        native_instant_vs_qpos_disagreements=native_disagreements,
        action_clipped_components=sum(row["action_clip_components"] for row in trace),
        action_component_samples=int(actions.size),
        action_clip_fraction=float(np.mean(np.abs(actions) > 1)),
        maximum_absolute_normalized_action=float(np.max(np.abs(actions))),
        raw_action_min_per_axis=np.min(actions, axis=0).tolist(),
        raw_action_max_per_axis=np.max(actions, axis=0).tolist(),
        trace=trace,
    )
    return result, initial_qpos, initial_mocap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/plug_analysis/randomized_feasibility_native.json"),
    )
    args = parser.parse_args()
    source_root = Path(native.__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("Use the frozen11a591f source via PYTHONPATH; native source differs")
    if subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise ValueError("Frozen tracked source must be clean")
    torch.set_num_threads(1)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    rows = []
    models = {}
    paired = {}
    configs = {}
    for variant in plug.VARIANTS:
        cfg = Experiment(
            task="plug",
            condition="wrist_static",
            occlusion="clean",
            num_envs=1,
            seed=SEEDS[0],
            randomize=True,
            plug_variant=variant,
            render_sensors=False,
            fixed_position=static_candidates("plug")[7],
            success_state_sample="current_qpos",
        )
        env = NativeEnv(cfg)
        configs[variant] = cfg.to_dict()
        models[variant] = {
            "mjb_sha256": model_sha(env.model),
            "nq": env.model.nq,
            "nv": env.model.nv,
            "nu": env.model.nu,
            "object_freejoint_qpos_address": int(env.object_adr),
        }
        for seed in SEEDS:
            row, qpos, mocap = rollout(env, seed)
            rows.append(row)
            paired.setdefault(seed, []).append((variant, qpos, mocap))
        print(
            variant,
            "held_success",
            sum(r["success"] for r in rows if r["variant"] == variant),
            "/",
            len(SEEDS),
            flush=True,
        )
    by_variant = {}
    for variant in plug.VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        times = [row["earliest_held_time_s"] for row in subset if row["success"]]
        by_variant[variant] = {
            "episodes": len(subset),
            "successes": len(times),
            "success_rate": len(times) / len(subset),
            "held_before_nominal_horizon": sum(
                row["held_before_nominal_horizon"] for row in subset
            ),
            "held_success_time_min_s": min(times) if times else None,
            "held_success_time_max_s": max(times) if times else None,
            "mean_terminal_error_mm": float(np.mean([row["terminal_error_mm"] for row in subset]))
            if all(row["terminal_error_mm"] is not None for row in subset)
            else None,
            "failed_seeds": [row["seed"] for row in subset if not row["success"]],
        }
    pairing = {
        str(seed): {
            "max_initial_qpos_difference": max(float(np.max(np.abs(q - q0))) for _, q, _ in states),
            "max_initial_mocap_difference": max(
                float(np.max(np.abs(m - m0))) for _, _, m in states
            ),
        }
        for seed, states in paired.items()
        for _, q0, m0 in states[:1]
    }
    modules = [
        "config.py",
        "native.py",
        "plug.py",
        "scenes.py",
        "robots/plug_control.py",
        "robots/kinematics.py",
        "occlusion.py",
    ]
    result = {
        "status": "COMPLETED",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - timer,
        "source_root": str(source_root),
        "source_revision": revision,
        "script_sha256": sha(__file__),
        "source_files_sha256": {
            name: sha(source_root / "src/active_perception_arms" / name) for name in modules
        },
        "packages": {name: version(name) for name in ("mujoco", "numpy", "torch")},
        "backend": "native MuJoCo CPU; float64 PlugIK; no renderer, images or learner",
        "controller": "Existing ScriptedPolicy unchanged; privileged object/goal/variant access",
        "configs": configs,
        "models": models,
        "seeds": list(SEEDS),
        "scoring": {
            "body_position": "Current raw object freejoint qpos xyz after every native step",
            "goal": "Current world fixture/goal site minus hidden prong xy offset",
            "distance_threshold_m": plug.SUCCESS_DISTANCE,
            "comparison": "strictly_less_than",
            "consecutive_control_samples": HOLD_SAMPLES,
            "control_step_s": 0.04,
            "control_sample_hz": 25,
            "nominal_horizon_s": 3.5,
            "maximum_steps": math.ceil(3.5 / 0.04),
            "timeout_quantized_s": math.ceil(3.5 / 0.04) * 0.04,
            "horizon_note": "MjLab rounds3.5/0.04 up to88 control steps (3.52s); the separate before-nominal flag exposes any success after3.5s.",
            "termination": "Stop at first held-three success; finite failures run through timeout; reset sample excluded",
        },
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "per_variant": by_variant,
        "within_native_seed_pairing": pairing,
        "all_qpos_finite": all(row["finite_qpos_all_samples"] for row in rows),
        "all_success_holds_verified": all(
            len(row["hold_error_mm"]) == HOLD_SAMPLES
            and all(error < plug.SUCCESS_DISTANCE * 1000 for error in row["hold_error_mm"])
            for row in rows
            if row["success"]
        ),
        "limits": [
            "Positive results establish feasibility only for these sampled native initial states with privileged control.",
            "Native float64 IK/physics is not the Warp training backend; matching numeric seed labels does not pair native states to Warp states.",
            "No images or learned visual inference were tested; results do not establish visual learnability, all-world solvability or perception utility.",
            "Controller, physics, inspection pause, action bounds and horizon were not tuned.",
        ],
        "rollouts": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        "saved", args.output, "successes", result["successes"], "/", result["episodes"], flush=True
    )


if __name__ == "__main__":
    main()
