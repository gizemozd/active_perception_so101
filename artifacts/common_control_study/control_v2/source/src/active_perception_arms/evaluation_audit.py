"""Read termination-time state before automatic resets, without altering dynamics."""

import hashlib

import numpy as np
import torch

from . import mdp, plug, pushing, transfer


class TerminationAudit:
    """Observe the manager's existing compute call; never forward/step or draw RNG."""

    def __init__(self, env):
        self.env = env
        self.original_compute = env.termination_manager.compute
        self.object_qpos_address = int(env.sim.mj_model.joint("object/free").qposadr[0])
        self.last_three_errors = None
        self.last_three_angles = None
        self.transfer_history = None
        self.push_history = None
        self.snapshot = None
        env.termination_manager.compute = self.compute

    def reset_batch(self):
        self.last_three_errors = torch.full(
            (self.env.num_envs, 3), float("nan"), device=self.env.device
        )
        self.snapshot = None

        if getattr(mdp.state(self.env).cfg, "task_revision", None) == "push_v2":
            self.push_history = torch.zeros(
                self.env.num_envs, 3, dtype=torch.bool, device=self.env.device
            )

        if getattr(mdp.state(self.env).cfg, "plug_wrist_yaw", False):
            self.last_three_angles = torch.full_like(self.last_three_errors, float("nan"))
        if transfer.is_v2(mdp.state(self.env).cfg):
            self.transfer_history = torch.zeros(
                self.env.num_envs, transfer.HOLD_STEPS, dtype=torch.bool, device=self.env.device
            )

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
        tcp, obj, _, goal = mdp.positions(self.env)
        if transfer.is_v2(mdp.state(self.env).cfg):
            obj = self.env.scene["object"].data.root_link_pos_w
        error = (obj - goal).norm(dim=-1)
        qpos = self.env.sim.data.qpos
        raw_obj = qpos[:, self.object_qpos_address : self.object_qpos_address + 3]
        qpos_error = (raw_obj - goal).norm(dim=-1)
        cfg = mdp.state(self.env).cfg
        sampling = getattr(cfg, "success_state_sample", "derived_substep")
        manager_error = (
            qpos_error
            if sampling == "current_qpos"
            and (
                cfg.task == "plug"
                or transfer.is_v2(cfg)
                or getattr(cfg, "task_revision", None) == "push_v2"
            )
            else error
        )
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
        if getattr(cfg, "plug_wrist_yaw", False):
            angle = mdp.plug_orientation_error(self.env)
            self.last_three_angles = torch.cat(
                (self.last_three_angles[:, 1:], angle[:, None]), dim=1
            )
            self.snapshot.update(
                qpos_orientation_error_rad=angle.clone(),
                last_three_orientation_errors_rad=self.last_three_angles.clone(),
            )
        if getattr(cfg, "reward_profile", None) == "plug_staged":
            s = mdp.state(self.env)
            # History through the previous reward sample; this observer never
            # advances stages or changes the physical/scoring computation.
            self.snapshot.update(
                plug_raised_history=s.plug_raised.clone(),
                plug_aligned_history=s.plug_aligned.clone(),
            )
        if transfer.is_v2(cfg):
            f = transfer.warp_features(
                self.env, mdp.state(self.env), tcp, raw_obj, goal, update=False
            )
            self.transfer_history = torch.cat(
                (self.transfer_history[:, 1:], f["instantaneous_success"][:, None]), dim=1
            )
            self.snapshot.update(
                transfer_criterion_history=self.transfer_history.clone(),
                transfer_lifted=f["lifted"].clone(),
                transfer_released=f["released"].clone(),
                transfer_placed=f["placed"].clone(),
                transfer_settled=f["settled"].clone(),
                physical_tcp_position=tcp.clone(),
                physical_object_position=raw_obj.clone(),
                physical_goal_position=goal.clone(),
                hand_distance=f["hand_distance"].clone(),
                linear_speed=f["linear_speed"].clone(),
                angular_speed=f["angular_speed"].clone(),
            )
        if getattr(cfg, "task_revision", None) == "push_v2":
            address = mdp.state(self.env).object_dof_address
            speed = self.env.sim.data.qvel[:, address : address + 3].norm(dim=-1)
            xy = (raw_obj[:, :2] - goal[:, :2]).norm(dim=-1)
            z = (raw_obj[:, 2] - goal[:, 2]).abs()
            tool = pushing.tool_position(self.env, mdp.state(self.env))
            withdrawn = (tool - raw_obj).norm(dim=-1) > 0.025
            clear = pushing.warp_clear(self.env)
            valid = (
                (xy < cfg.push_success_tolerance) & (z < 0.004)
                & withdrawn & (speed < 0.035) & clear
            )
            self.push_history = torch.cat((self.push_history[:, 1:], valid[:, None]), dim=1)
            self.snapshot.update(
                push_criterion_history=self.push_history.clone(),
                physical_tcp_position=tcp.clone(),
                physical_object_position=raw_obj.clone(),
                physical_goal_position=goal.clone(),
                linear_speed=speed.clone(),
                push_xy_error=xy.clone(),
                push_z_error=z.clone(),
                push_withdrawn=withdrawn.clone(),
                push_contact_free=clear.clone(),
                physical_tool_position=tool.clone(),
            )
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
        "success_hold_violations": sum(
            r["success_hold_steps"]
            < (transfer.HOLD_STEPS if "transfer_criterion_history" in r else 3)
            for r in successes
        ),
        "success_transfer_violations": sum(
            not all(r["transfer_criterion_history"]) for r in successes
        )
        if records and "transfer_criterion_history" in records[0]
        else None,
        "success_push_violations": sum(not all(r["push_criterion_history"]) for r in successes)
        if records and "push_criterion_history" in records[0]
        else None,
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
        "success_orientation_violations": sum(
            any(
                not np.isfinite(v) or v >= plug.SUCCESS_ANGLE
                for v in r["last_three_orientation_errors_rad"]
            )
            for r in successes
        )
        if records and "last_three_orientation_errors_rad" in records[0]
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
