"""Read termination-time state before automatic resets, without altering dynamics."""

import hashlib

import numpy as np
import torch

from . import mdp, plug


class TerminationAudit:
    """Observe the manager's existing compute call; never forward/step or draw RNG."""

    def __init__(self, env):
        self.env = env
        self.original_compute = env.termination_manager.compute
        self.object_qpos_address = int(env.sim.mj_model.joint("object/free").qposadr[0])
        self.last_three_errors = None
        self.snapshot = None
        env.termination_manager.compute = self.compute

    def reset_batch(self):
        self.last_three_errors = torch.full(
            (self.env.num_envs, 3), float("nan"), device=self.env.device
        )
        self.snapshot = None

    def initial_state_hashes(self, observations, sensors):
        """World-indexed hashes support repeat checks; origins remain part of state."""
        arrays = [
            self.env.sim.data.qpos.cpu().numpy(),
            self.env.sim.data.qvel.cpu().numpy(),
            self.env.sim.data.mocap_pos.cpu().numpy(),
            *[observations[name].cpu().numpy() for name in sensors],
        ]
        return [
            hashlib.sha256(
                b"".join(np.ascontiguousarray(a[i]).tobytes() for a in arrays)
            ).hexdigest()
            for i in range(self.env.num_envs)
        ]

    def initial_hash_components(self, observations, sensors):
        arrays = [
            self.env.sim.data.qpos.cpu().numpy(),
            self.env.sim.data.qvel.cpu().numpy(),
            self.env.sim.data.mocap_pos.cpu().numpy(),
        ]
        physics = [
            hashlib.sha256(
                b"".join(np.ascontiguousarray(a[i]).tobytes() for a in arrays)
            ).hexdigest()
            for i in range(self.env.num_envs)
        ]
        images = {
            name: [
                hashlib.sha256(
                    np.ascontiguousarray(observations[name][i].cpu().numpy()).tobytes()
                ).hexdigest()
                for i in range(self.env.num_envs)
            ]
            for name in sensors
        }
        return physics, images

    def compute(self):
        # This is exactly where the installed MjLab environment already computes
        # success. Reading here avoids post-reset state and the later forward call.
        done = self.original_compute()
        _, obj, _, goal = mdp.positions(self.env)
        error = (obj - goal).norm(dim=-1)
        qpos = self.env.sim.data.qpos
        raw_obj = qpos[:, self.object_qpos_address : self.object_qpos_address + 3]
        qpos_error = (raw_obj - goal).norm(dim=-1)
        cfg = mdp.state(self.env).cfg
        sampling = getattr(cfg, "success_state_sample", "derived_substep")
        manager_error = qpos_error if sampling == "current_qpos" and cfg.task == "plug" else error
        self.last_three_errors = torch.cat(
            (self.last_three_errors[:, 1:], manager_error[:, None]), dim=1
        )
        self.snapshot = {
            "manager_position_error_m": manager_error.clone(),
            "derived_position_error_m": error.clone(),
            "qpos_position_error_m": qpos_error.clone(),
            "derived_to_qpos_displacement_m": (raw_obj - obj).norm(dim=-1).clone(),
            "success_hold_steps": mdp.state(self.env).hold.clone(),
            "last_three_manager_errors_m": self.last_three_errors.clone(),
            "success": self.env.termination_manager.get_term("success").clone(),
            "failure": self.env.termination_manager.get_term("failure").clone(),
            "timeout": self.env.termination_manager.time_outs.clone(),
        }
        return done

    def close(self):
        self.env.termination_manager.compute = self.original_compute


def summarize_termination_audit(records, task):
    successes = [r for r in records if r["success"]]
    return {
        "sampling": "After termination_manager.compute(), before reward/reset/forward; no physics or random-state changes.",
        "episodes": len(records),
        "terminal_flags": {
            flag: sum(bool(r[flag]) for r in records) for flag in ("success", "failure", "timeout")
        },
        "max_derived_to_qpos_displacement_m": max(
            (r["derived_to_qpos_displacement_m"] for r in records), default=None
        ),
        "success_hold_violations": sum(r["success_hold_steps"] < 3 for r in successes),
        "success_distance_violations": sum(
            r["manager_position_error_m"] >= plug.SUCCESS_DISTANCE for r in successes
        )
        if task == "plug"
        else None,
        "success_three_sample_violations": sum(
            any(
                not np.isfinite(v) or v >= plug.SUCCESS_DISTANCE
                for v in r["last_three_manager_errors_m"]
            )
            for r in successes
        )
        if task == "plug"
        else None,
        "success_qpos_above_distance_threshold": sum(
            r["qpos_position_error_m"] >= plug.SUCCESS_DISTANCE for r in successes
        )
        if task == "plug"
        else None,
        "mean_terminal_position_error_m": float(
            np.mean([r["manager_position_error_m"] for r in records])
        )
        if records
        else None,
        "mean_success_terminal_position_error_m": float(
            np.mean([r["manager_position_error_m"] for r in successes])
        )
        if successes
        else None,
        "records": records,
    }
