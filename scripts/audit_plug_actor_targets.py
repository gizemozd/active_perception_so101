#!/usr/bin/env python3
"""Conditional frozen-actor target/feature audit; no learner or controller changes.

Import clean frozen11a591f source through PYTHONPATH. Requires an explicit final
checkpoint and its512episode physical evaluation reference. Passive hooks observe
the ONE deterministic actor forward per control step. Raw arrays remain ignored
NPZ artifacts; JSON/CSV contain summaries and provenance only.
"""

import argparse
import csv
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

import mujoco
import numpy as np
import torch

from active_perception_arms import mdp, plug, policy
from active_perception_arms.config import saved_experiment, static_candidates
from active_perception_arms.environment import make_env
from active_perception_arms.evaluate import load_actor
from active_perception_arms.evaluation_audit import TerminationAudit, summarize_termination_audit

SOURCE_REVISION = "11a591f7a2129484d555caf115fcdbf1a709a0c9"
SEEDS = tuple(range(10000, 10004))
NUM_ENVS = 128
INSPECTION_STEP = 24
TIME_EDGES = (0.0, 1.0, 1.5, 2.0, 2.5, 3.52)


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
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class PassiveFeatures:
    """Observe consumed module outputs; never call a model or mutate its state."""

    def __init__(self, actor, sensors):
        if not actor.is_recurrent or set(actor.cnns) != set(sensors):
            raise ValueError("Diagnostic expects the saved dual-view recurrent actor")
        self.handles = []
        self.expected = ("actor", "normalized_proprio", "gru", *[f"cnn_{s}" for s in sensors])
        self.capture = False
        self.counts = {}
        self.outputs = {}
        modules = {
            "actor": actor,
            "normalized_proprio": actor.obs_normalizer,
            "gru": actor.rnn,
            **{f"cnn_{s}": actor.cnns[s] for s in sensors},
        }
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def observe(module, inputs, output):
            self.counts[name] = self.counts.get(name, 0) + 1
            if self.capture:
                self.outputs[name] = output.detach().clone()
            # Returning None leaves the real forward output unchanged.

        return observe

    def begin(self, capture=False):
        self.capture = capture
        self.counts = {}
        self.outputs = {}

    def end(self):
        if self.counts != dict.fromkeys(self.expected, 1):
            raise RuntimeError(f"Single-forward invariant violated: {self.counts}")
        return self.outputs

    def close(self):
        for handle in self.handles:
            handle.remove()


class PhysicalAudit(TerminationAudit):
    """Snapshot action target and physical state before manager autoresets."""

    def compute(self):
        done = super().compute()
        tcp, _, _, goal = mdp.positions(self.env)
        arms = self.env.action_manager.get_term("arms")
        address = self.object_qpos_address
        self.snapshot.update(
            body_qpos_world_m=self.env.sim.data.qpos[:, address : address + 3].clone(),
            body_goal_world_m=goal.clone(),
            tcp_derived_world_m=tcp.clone(),
            processed_tcp_target_local_m=arms.tcp_target.clone(),
            processed_normalized_action=arms.raw_action.clone(),
            finite_qpos=torch.isfinite(self.env.sim.data.qpos).all(dim=-1).clone(),
            episode_step=self.env.episode_length_buf.clone(),
        )
        return done


def concatenate(parts):
    return {name: np.concatenate(values, axis=0) for name, values in parts.items()}


def vector_summary(values):
    norms = np.linalg.norm(values, axis=-1)
    return {
        "mean_signed_mm": (values.mean(axis=0) * 1000).tolist(),
        "std_signed_mm": (values.std(axis=0) * 1000).tolist(),
        "median_norm_mm": float(np.median(norms) * 1000),
        "p90_norm_mm": float(np.quantile(norms, 0.9) * 1000),
    }


def summarize_trace(trace):
    """Command residuals are phase-dependent, not direct perception estimates."""
    rows = []
    for variant_id, variant in enumerate(plug.VARIANTS):
        for lower, upper in zip(TIME_EDGES[:-1], TIME_EDGES[1:], strict=True):
            selected = (
                (trace["variant_id"] == variant_id)
                & (trace["action_time_s"] >= lower)
                & (trace["action_time_s"] < upper)
            )
            if not selected.any():
                continue
            wanted = trace["final_required_tcp_local_m"][selected]
            actual = trace["tcp_derived_local_m"][selected]
            processed = trace["processed_tcp_target_local_m"][selected]
            raw = trace["raw_normalized_mean"][selected]
            offsets = np.array([plug.OFFSETS[v] for v in plug.VARIANTS])
            socket_xy = wanted[:, :2] + offsets[variant_id] - np.array(plug.GRIP_OFFSET[:2])
            candidates = socket_xy[:, None, :] - offsets[None] + np.array(plug.GRIP_OFFSET[:2])
            nearest = np.linalg.norm(processed[:, None, :2] - candidates, axis=-1).argmin(1)
            descended = trace["body_qpos_local_m"][selected, 2] < 0.03
            rows.append(
                {
                    "variant": variant,
                    "action_time_bin_s": [lower, upper],
                    "poststep_body_time_offset_s": 0.04,
                    "step_samples": int(selected.sum()),
                    "episodes": int(np.unique(trace["episode"][selected]).size),
                    "raw_mean_request_vs_final_tcp": vector_summary(
                        trace["raw_absolute_tcp_request_local_m"][selected] - wanted
                    ),
                    "clipped_request_vs_final_tcp": vector_summary(
                        trace["clipped_tcp_request_local_m"][selected] - wanted
                    ),
                    "processed_target_vs_final_tcp": vector_summary(processed - wanted),
                    "derived_tcp_tracking": vector_summary(actual - processed),
                    "current_body_vs_final_body": vector_summary(
                        trace["body_qpos_local_m"][selected] - trace["body_goal_local_m"][selected]
                    ),
                    "raw_mean_outside_box_component_fraction": float((np.abs(raw) > 1).mean()),
                    "target_xy_within2mm_fraction": float(
                        np.mean(np.linalg.norm((processed - wanted)[:, :2], axis=-1) < 0.002)
                    ),
                    "body_xy_within2mm_fraction": float(
                        np.mean(
                            np.linalg.norm(trace["body_signed_error_m"][selected, :2], axis=-1)
                            < 0.002
                        )
                    ),
                    "body_below30mm_fraction": float(
                        np.mean(trace["body_qpos_local_m"][selected, 2] < 0.03)
                    ),
                    "processed_target_nearest_final_variant_xy_counts": np.bincount(
                        nearest, minlength=4
                    ).tolist(),
                    "descended_samples": int(descended.sum()),
                    "descended_processed_target_nearest_final_variant_xy_counts": np.bincount(
                        nearest[descended], minlength=4
                    ).tolist(),
                }
            )
    return rows


def ridge_prediction(train_x, train_y, test_x, alpha=1e-3):
    """Fixed ridge; center/scale and target intercept fitted on training only."""
    center, scale = train_x.mean(0), train_x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = (train_x - center) / scale
    y_center = train_y.mean(0)
    weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (train_y - y_center))
    return (test_x - center) / scale @ weights + y_center


def probe_metrics(predicted, target):
    error = np.linalg.norm(predicted[:, :2] - target[:, :2], axis=-1)
    actual_class = target[:, 2:].argmax(1)
    predicted_class = predicted[:, 2:].argmax(1)
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (actual_class, predicted_class), 1)
    return {
        "episodes": len(target),
        "socket_xy_rmse_euclidean_mm": float(np.sqrt(np.mean(error**2)) * 1000),
        "socket_xy_median_error_mm": float(np.median(error) * 1000),
        "socket_xy_p90_error_mm": float(np.quantile(error, 0.9) * 1000),
        "socket_xy_within2mm_fraction": float(np.mean(error < 0.002)),
        "variant_accuracy": float(np.mean(actual_class == predicted_class)),
        "joint_variant_correct_and_socket_within2mm_fraction": float(
            np.mean((actual_class == predicted_class) & (error < 0.002))
        ),
        "variant_confusion_true_rows_predicted_columns": confusion.tolist(),
    }


def accessibility_probe(inspection, sensors):
    cnn = np.concatenate([inspection[f"cnn_{s}"] for s in sensors], axis=1).astype(np.float64)
    proprio = inspection["proprio"].astype(np.float64)
    target = np.column_stack(
        [inspection["socket_goal_local_m"][:, :2], np.eye(4)[inspection["variant_id"]]]
    )
    results = []
    for name, x in (
        ("cnn_only", cnn),
        ("proprio_only", proprio),
        ("cnn_plus_proprio", np.column_stack([cnn, proprio])),
        ("gru_only", inspection["gru"].astype(np.float64)),
    ):
        for seed in SEEDS:
            test = inspection["reset_seed"] == seed
            train = ~test
            if not test.any() or not train.any():
                raise ValueError("All four reset seed groups required for probe")
            prediction = ridge_prediction(x[train], target[train], x[test])
            baseline = np.broadcast_to(target[train].mean(0), target[test].shape)
            results.append(
                {
                    "features": name,
                    "heldout_reset_seed": seed,
                    "training_episodes": int(train.sum()),
                    "feature_dimensions": x.shape[1],
                    "metrics": probe_metrics(prediction, target[test]),
                    "training_mean_baseline": probe_metrics(baseline, target[test]),
                }
            )
    return {
        "status": "OFFLINE_ACCESSIBILITY_PROBE_COMPLETED",
        "alpha": 1e-3,
        "formula": "sum squared standardized-feature training residual + alpha*L2(weights); unpenalized training-target intercept",
        "split": "Train three reset seeds; hold out fourth, four prespecified folds; train-only feature centering/scaling; no holdout tuning.",
        "labels": "Socket goal localXY (not corrected body goal) plus variantonehot, orderedxm/xp/ym/yp; privileged labels never enter actor inputs.",
        "folds": results,
        "limits": "Diagnostic decoder fitting is not actor/policy training. CNN and GRU groups describe linear accessibility in consumed encoder features versus recurrent state, not policy use or perception causality; absent linear decoding does not mean information is unavailable. These are within-protocol seed folds, not independent trained models or new-world guarantees.",
    }


def save_npz(path, arrays):
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "ignored_raw_artifact": True,
        "all_numeric_fields_finite": all(np.isfinite(v).all() for v in arrays.values()),
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
    }


def write_csv(path, summaries):
    rows = []
    for summary in summaries:
        base = {k: v for k, v in summary.items() if not isinstance(v, (list, dict))}
        base["bin_start_s"], base["bin_end_s"] = summary["action_time_bin_s"]
        for name, value in summary.items():
            if isinstance(value, dict):
                for key, stat in value.items():
                    if isinstance(stat, list):
                        for axis, item in zip("xyz", stat, strict=True):
                            base[f"{name}_{key}_{axis}"] = item
                    else:
                        base[f"{name}_{key}"] = stat
            elif isinstance(value, list) and name != "action_time_bin_s":
                for variant, count in zip(plug.VARIANTS, value, strict=True):
                    base[f"{name}_{variant}"] = count
        rows.append(base)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args, cfg, reference, provenance):
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Allocate one CUDA GPU for this Warp-only audit")
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    env = make_env(cfg, args.device)
    audit = PhysicalAudit(env)
    hooks = None
    records, trace_parts, inspection_parts = [], {}, {}
    combined_hashes, physics_hashes = [], []
    sensor_hashes = {s: [] for s in cfg.sensors}
    forward_calls, batch_steps = 0, []
    variants = np.array(plug.variant_assignment(NUM_ENVS), dtype=int)
    try:
        obs, _ = env.reset(seed=SEEDS[0])
        actor = load_actor(args.checkpoint, obs, args.device)
        base_model_sha256 = model_sha(env.sim.mj_model)
        before_actor = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        hooks = PassiveFeatures(actor, cfg.sensors)
        origins = env.scene.env_origins.detach().cpu().numpy()
        with torch.inference_mode():
            for batch, seed in enumerate(SEEDS):
                obs, _ = env.reset(seed=seed)
                audit.reset_batch()
                actor.reset()
                hashes = audit.initial_state_hashes(obs, cfg.sensors)
                physics, sensors = audit.initial_hash_components(obs, cfg.sensors)
                offset = batch * NUM_ENVS
                for name, actual, expected in (
                    ("combined", hashes, reference["initial_state_sha256"]),
                    ("physics", physics, reference["initial_physics_state_sha256"]),
                    *[(s, sensors[s], reference["initial_sensor_sha256"][s]) for s in cfg.sensors],
                ):
                    if actual != expected[offset : offset + NUM_ENVS]:
                        raise RuntimeError(f"Strict initial pairing failed: batch{batch}/{name}")
                combined_hashes.extend(hashes)
                physics_hashes.extend(physics)
                for sensor, values in sensors.items():
                    sensor_hashes[sensor].extend(values)
                finished = np.zeros(NUM_ENVS, dtype=bool)
                finite_all = np.ones(NUM_ENVS, dtype=bool)
                minimum = np.full(NUM_ENVS, np.inf)
                batch_records = [None] * NUM_ENVS
                for step in range(math.ceil(cfg.episode_seconds / cfg.step_dt)):
                    live = ~finished
                    ids = np.flatnonzero(live)
                    capture = step == INSPECTION_STEP
                    inspection = None
                    if capture:
                        _, _, _, goal = mdp.positions(env)
                        body_goal = goal.detach().cpu().numpy() - origins
                        inspection = {
                            "episode": offset + ids,
                            "reset_seed": np.full(len(ids), seed),
                            "world": ids,
                            "variant_id": variants[ids],
                            "time_s": np.full(len(ids), step * cfg.step_dt),
                            "proprio": obs["proprio"].detach().cpu().numpy()[ids].copy(),
                            "body_goal_local_m": body_goal[ids],
                            "socket_goal_local_m": body_goal[ids]
                            + np.column_stack(
                                [
                                    [plug.OFFSETS[plug.VARIANTS[v]][axis] for v in variants[ids]]
                                    for axis in range(2)
                                ]
                                + [np.zeros(len(ids))]
                            ),
                            **{s: obs[s].detach().cpu().numpy()[ids].copy() for s in cfg.sensors},
                        }
                    hooks.begin(capture=capture)
                    # EXACTLY ONE actor call. Default deterministic mean, as evaluate.py.
                    action = actor(obs).clone()
                    features = hooks.end()
                    forward_calls += 1
                    if inspection is not None:
                        for name, value in features.items():
                            if name == "gru":
                                value = value.squeeze(0)
                            inspection[name] = value.detach().cpu().numpy()[ids].copy()
                        for name, value in inspection.items():
                            inspection_parts.setdefault(name, []).append(value)
                    raw = action.detach().cpu().numpy()
                    if not np.isfinite(raw).all():
                        raise RuntimeError("Actor emitted nonfinite mean actions")
                    obs, _, terminated, truncated, _ = env.step(action)
                    snap = {k: v.detach().cpu().numpy() for k, v in audit.snapshot.items()}
                    low, high = np.array(plug.ACTION_LOW), np.array(plug.ACTION_HIGH)
                    raw_request = low + (raw[:, :3] + 1) * 0.5 * (high - low)
                    clipped = np.clip(raw[:, :3], -1, 1)
                    clipped_request = low + (clipped + 1) * 0.5 * (high - low)
                    if not np.array_equal(snap["processed_normalized_action"], np.clip(raw, -1, 1)):
                        raise RuntimeError(
                            "Recorded processed action differs from real action term"
                        )
                    body = snap["body_qpos_world_m"] - origins
                    goal = snap["body_goal_world_m"] - origins
                    wanted_tcp = goal + np.array(plug.GRIP_OFFSET)
                    row_arrays = {
                        "episode": offset + ids,
                        "reset_seed": np.full(len(ids), seed),
                        "world": ids,
                        "variant_id": variants[ids],
                        "step": np.full(len(ids), step),
                        "action_time_s": np.full(len(ids), step * cfg.step_dt),
                        "physical_poststep_time_s": np.full(len(ids), (step + 1) * cfg.step_dt),
                        "raw_normalized_mean": raw[ids],
                        "raw_absolute_tcp_request_local_m": raw_request[ids],
                        "clipped_tcp_request_local_m": clipped_request[ids],
                        "processed_tcp_target_local_m": snap["processed_tcp_target_local_m"][ids],
                        "tcp_derived_local_m": snap["tcp_derived_world_m"][ids] - origins[ids],
                        "body_qpos_local_m": body[ids],
                        "body_goal_local_m": goal[ids],
                        "final_required_tcp_local_m": wanted_tcp[ids],
                        "body_signed_error_m": body[ids] - goal[ids],
                        "body_error_m": snap["qpos_position_error_m"][ids],
                        "hold_steps": snap["success_hold_steps"][ids],
                    }
                    for name, value in row_arrays.items():
                        trace_parts.setdefault(name, []).append(value.copy())
                    minimum[live] = np.minimum(minimum[live], snap["qpos_position_error_m"][live])
                    finite_all[live] &= snap["finite_qpos"][live]
                    done = (terminated | truncated).detach().cpu().numpy()
                    for i in np.flatnonzero(done & live):
                        batch_records[i] = {
                            "episode": offset + int(i),
                            "reset_seed": seed,
                            "world": int(i),
                            "variant": plug.VARIANTS[variants[i]],
                            **{k: v[i].tolist() for k, v in snap.items()},
                            "elapsed_s": (step + 1) * cfg.step_dt,
                            "minimum_current_body_error_m": float(minimum[i]),
                            "finite_qpos_all_preterminal_samples": bool(finite_all[i]),
                            "raw_normalized_mean": raw[i].tolist(),
                            "raw_absolute_tcp_request_local_m": raw_request[i].tolist(),
                            "clipped_tcp_request_local_m": clipped_request[i].tolist(),
                            "body_qpos_local_m": body[i].tolist(),
                            "body_goal_local_m": goal[i].tolist(),
                            "final_required_tcp_local_m": wanted_tcp[i].tolist(),
                            "derived_tcp_tracking_signed_error_m": (
                                snap["tcp_derived_world_m"][i]
                                - origins[i]
                                - snap["processed_tcp_target_local_m"][i]
                            ).tolist(),
                        }
                    finished |= done
                    actor.reset(terminated | truncated)
                    if finished.all():
                        break
                batch_steps.append(step + 1)
                if any(row is None for row in batch_records):
                    raise RuntimeError("A first episode lacked a terminal record at timeout")
                records.extend(batch_records)
                print(
                    f"batch{batch} seed{seed} successes{sum(r['success'] for r in batch_records)}/128",
                    flush=True,
                )
        actor_unchanged = all(
            torch.equal(v, before_actor[k]) for k, v in actor.state_dict().items()
        )
        if not actor_unchanged:
            raise RuntimeError("Actor parameters/normalization buffers changed during audit")
        if sha(args.checkpoint) != provenance["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed during read-only evaluation")
        if sha(args.reference_report) != provenance["reference_report_sha256"]:
            raise RuntimeError("Reference changed during read-only evaluation")
    finally:
        if hooks:
            hooks.close()
        audit.close()
        env.close()
    trace, inspection = concatenate(trace_parts), concatenate(inspection_parts)
    summaries = summarize_trace(trace)
    physical = summarize_termination_audit(records, "plug")
    physical.pop("records")
    trace_path = args.output.with_suffix(".trace.npz")
    inspection_path = args.output.with_suffix(".inspection.npz")
    csv_path = args.output.with_suffix(".csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "COMPLETED",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - timer,
        **provenance,
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": platform.node(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device),
            "packages": {
                n: version(n)
                for n in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib")
            },
        },
        "experiment": cfg.to_dict(),
        "warp_host_base_model_mjb_sha256": base_model_sha256,
        "model_hash_note": "Host base-model MJB does not encode every perworld variant patch; frozen scene/plug source hashes and explicit balanced assignment are recorded separately.",
        "seed": SEEDS[0],
        "reset_seeds": list(SEEDS),
        "episodes": len(records),
        "successes": sum(r["success"] for r in records),
        "per_variant": {
            v: {
                "episodes": sum(r["variant"] == v for r in records),
                "successes": sum(r["success"] for r in records if r["variant"] == v),
            }
            for v in plug.VARIANTS
        },
        "first_terminal_records": records,
        "termination_audit": physical,
        "initial_state_sha256": combined_hashes,
        "initial_physics_state_sha256": physics_hashes,
        "initial_sensor_sha256": sensor_hashes,
        "strict_initial_pairing": {
            "passed": True,
            "episodes": len(records),
            "combined_mismatches": 0,
            "physics_mismatches": 0,
            "sensor_mismatches": dict.fromkeys(cfg.sensors, 0),
        },
        "single_forward_invariant": {
            "actor_calls": forward_calls,
            "expected_calls": sum(batch_steps),
            "batch_control_steps": batch_steps,
            "all_consumed_modules_called_once_per_step": True,
            "parameters_and_normalization_buffers_unchanged": actor_unchanged,
            "learner_updates": 0,
        },
        "command_error_label": "Signed phase-conditioned residual against FINAL required TCP goal, not a direct actor perception estimate. Approaches and descent legitimately use intermediate targets. Nearest-final-variant target counts likewise describe commands, not classified perception; candidate orderxm/xp/ym/yp, with descended samples reported separately. All trace rows are live first episodes; no postterminal reset rows.",
        "physical_error_label": "Rawfreejoint current body-goal distance is authoritative. Derived TCP tracking may lag one physics substep; all vectors use local origins and the verified bodygoal+(0,-.0015,.045) TCP convention.",
        "per_variant_time_bin_summaries": summaries,
        "inspection_snapshot": {
            "time_s": INSPECTION_STEP * cfg.step_dt,
            "before_actor_forward": True,
            "episodes": len(inspection["episode"]),
            "cnn_sensor_order": list(cfg.sensors),
            "features_from_that_single_forward": True,
        },
        "raw_artifacts": {
            "trace": save_npz(trace_path, trace),
            "inspection": save_npz(inspection_path, inspection),
        },
        "accessibility_probe": accessibility_probe(inspection, cfg.sensors)
        if args.probe
        else {"status": "NOT_REQUESTED"},
        "csv": str(csv_path),
        "limits": [
            "A single frozen learned policy diagnostic, not active-perception utility or policy training.",
            "Strict initial pairing does not guarantee identical later trajectories; numerical evaluation repeat variation remains possible.",
            "Time-bin samples are episode-correlated and their signed command residuals are not visual-localization ground truth.",
            "Linear accessibility probes cannot establish policy use, representation sufficiency or perception causality; no holdout hyperparameter selection is performed.",
        ],
    }
    write_csv(csv_path, summaries)
    result["csv_sha256"] = sha(csv_path)
    with args.output.open("x") as stream:
        json.dump(json_safe(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": len(records),
                "successes": result["successes"],
                "wall_seconds": result["wall_seconds"],
            }
        ),
        flush=True,
    )


def prepare(args):
    source = Path(policy.__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if revision != SOURCE_REVISION or dirty:
        raise ValueError("Import clean frozen11a591f source via PYTHONPATH")
    original = saved_experiment(
        json.loads(args.checkpoint.with_name("experiment.json").read_text())
    )
    runner = json.loads(args.checkpoint.with_name("runner.json").read_text())
    if not (
        original.task == "plug"
        and original.condition == "wrist_static"
        and original.occlusion == "clean"
        and original.memory == "gru"
        and original.randomize
        and original.render_sensors
        and original.plug_variant is None
        and original.success_state_sample == "current_qpos"
        and np.allclose(original.fixed_position, static_candidates("plug")[7], atol=1e-12)
        and original.step_dt == 0.04
        and original.initial_seconds == 1.0
        and original.episode_seconds == 3.5
    ):
        raise ValueError("Expected clean fixedview7/currentqpos randomized plug GRU checkpoint")
    if (
        args.checkpoint.name != "model_750.pt"
        or runner["max_iterations"] != 751
        or runner["num_steps_per_env"] != 24
    ):
        raise ValueError("Use the explicit final751 checkpoint; no midpoint selection")
    cfg = replace(original, num_envs=NUM_ENVS, seed=SEEDS[0])
    reference = json.loads(args.reference_report.read_text())
    reference_cfg = saved_experiment(reference["experiment"])
    if json.loads(json.dumps(cfg.to_dict())) != json.loads(json.dumps(reference_cfg.to_dict())):
        raise ValueError("Reference experiment differs from actual diagnostic configuration")
    if (
        Path(reference["checkpoint"]).resolve() != args.checkpoint.resolve()
        or reference["seed"] != SEEDS[0]
        or reference["episodes"] != 512
    ):
        raise ValueError(
            "Require same explicit checkpoint and512balanced seeded physical reference"
        )
    expected_variants = [plug.VARIANTS[v] for v in plug.variant_assignment(NUM_ENVS)] * len(SEEDS)
    if reference["variants"] != expected_variants:
        raise ValueError("Reference first episodes are not variant-balanced in world order")
    for field in (
        "success_hold_violations",
        "success_distance_violations",
        "success_three_sample_violations",
        "success_qpos_above_distance_threshold",
    ):
        if reference["termination_audit"][field] != 0:
            raise ValueError("Reference contains physical success criterion violations")
    for key in ("freeze_camera_after", "hold_external_after", "camera_trace_in"):
        if reference["interventions"].get(key) is not None:
            raise ValueError("Reference has camera interventions")
    if reference["interventions"].get("reset_memory"):
        raise ValueError("Reference has memory intervention")
    for hashes in (
        reference["initial_state_sha256"],
        reference["initial_physics_state_sha256"],
        *reference["initial_sensor_sha256"].values(),
    ):
        if len(hashes) != 512:
            raise ValueError("Reference lacks512split initial hashes")
    checkpoint_sha = sha(args.checkpoint)
    if not args.dry_run and not args.expected_checkpoint_sha256:
        raise ValueError("Freeze the checkpoint SHA256 explicitly for execution")
    if args.expected_checkpoint_sha256 and args.expected_checkpoint_sha256 != checkpoint_sha:
        raise ValueError("Checkpoint changed from frozen submission hash")
    data_root = Path(__file__).resolve().parents[1]
    args.output.resolve().relative_to(data_root / "artifacts")
    outputs = [
        args.output,
        args.output.with_suffix(".csv"),
        args.output.with_suffix(".trace.npz"),
        args.output.with_suffix(".inspection.npz"),
    ]
    if any(path.exists() for path in outputs):
        raise ValueError("An output exists; preserve results and choose a new stem")
    for path in outputs[-2:]:
        ignored = subprocess.run(
            ["git", "-C", str(data_root), "check-ignore", "-q", str(path.resolve())], check=False
        )
        if ignored.returncode != 0:
            raise ValueError("RawNPZ must be gitignored")
    modules = (
        "config.py",
        "mdp.py",
        "plug.py",
        "policy.py",
        "environment.py",
        "evaluation_audit.py",
        "evaluate.py",
        "robots/plug_control.py",
        "robots/kinematics.py",
        "scenes.py",
    )
    provenance = {
        "source_root": str(source),
        "source_revision": revision,
        "helper_sha256": sha(__file__),
        "launcher_sha256": sha(data_root / "scripts/slurm/audit_plug_actor_targets.sbatch"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "runner_sha256": sha(args.checkpoint.with_name("runner.json")),
        "saved_experiment_sha256": sha(args.checkpoint.with_name("experiment.json")),
        "reference_report": str(args.reference_report.resolve()),
        "reference_report_sha256": sha(args.reference_report),
        "source_files_sha256": {p: sha(source / "src/active_perception_arms" / p) for p in modules},
        "conditional_entry": "Root must finish/audit fresh751 and confirm competence gate failure before submission. Helper validates finalcheckpoint/reference/protocol; it does not itself declare the training experiment complete.",
    }
    return cfg, reference, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts/plug_analysis/target_diagnostic/report.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Fixed offline linear accessibility probe, no actor training",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg, reference, provenance = prepare(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "PREPARED_NOT_SUBMITTED",
                    "experiment": cfg.to_dict(),
                    "episodes": 512,
                    "inspection_time_s": 0.96,
                    "probe_requested": args.probe,
                    **provenance,
                },
                indent=2,
            )
        )
        return
    torch.set_num_threads(1)
    run(args, cfg, reference, provenance)


if __name__ == "__main__":
    main()
