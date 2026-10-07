"""Vectorized task logic; plug IK runs once per control step, without host transfers."""

from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from . import plug
from .config import SOURCE_XY, TARGET_XY, Experiment
from .native import calibration
from .robots.kinematics import quat_wxyz_to_mat
from .robots.plug_control import PlugIK
from .scenes import grasp_relpose


class State:
    def __init__(self, env, cfg):
        self.cfg = cfg
        self.home = {
            k: torch.tensor(v, dtype=torch.float32, device=env.device)
            for k, v in calibration(cfg.task).items()
        }
        n, device = env.num_envs, env.device
        self.onset = torch.zeros(n, device=device)
        self.duration = torch.zeros(n, device=device)
        self.side = torch.ones(n, device=device)
        self.hold = torch.zeros(n, dtype=torch.long, device=device)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=device)
        self.previous_potential = torch.zeros(n, device=device)
        self.potential_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_success_step = -1
        self.ids = {
            name: env.sim.mj_model.site(name).id
            for name in (
                "manipulator/gripperframe",
                "object/tip",
                "fixture/goal",
                "camera_arm/wrist_cam_site",
            )
        }
        self.object_id = env.sim.mj_model.body("object/object").id
        self.env_ids = torch.arange(n, device=device)
        self.fixed_calibration = torch.tensor(
            (*cfg.fixed_position, *cfg.fixed_lookat), device=device
        )
        self.panel_pose = torch.zeros(n, 7, device=device)
        self.plug_offsets = torch.tensor(
            [plug.OFFSETS[plug.VARIANTS[i]] for i in plug.variant_assignment(n, cfg.plug_variant)],
            device=device,
        )


def state(env, cfg=None):
    if not hasattr(env, "task_state"):
        if cfg is None:
            cfg = env.action_manager.get_term("arms").cfg.experiment
        env.task_state = State(env, cfg)
    return env.task_state


def positions(env):
    s = state(env)
    sites = env.sim.data.site_xpos
    goal = sites[:, s.ids["fixture/goal"]]
    if s.cfg.task == "plug":
        goal = goal.clone()
        goal[:, :2] -= s.plug_offsets
    return (
        sites[:, s.ids["manipulator/gripperframe"]],
        env.scene["object"].data.root_link_pos_w,
        sites[:, s.ids["object/tip"]],
        goal,
    )


def reset_task(env, env_ids, cfg):
    s = state(env, cfg)
    ids = env_ids
    n = len(ids)
    origins = env.scene.env_origins[ids]
    for name in ("manipulator", "camera_arm"):
        q = s.home[name].expand(n, -1)
        env.scene[name].write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        env.scene[name].set_joint_position_target(q, env_ids=ids)
    jitter = 0.008 if cfg.randomize else 0.0
    target_xy = TARGET_XY if cfg.task != "push" else (-0.025, 0.055)
    fixture = torch.zeros(n, 7, device=env.device)
    fixture[:, :2] = (
        torch.tensor(target_xy, device=env.device)
        + (torch.rand(n, 2, device=env.device) * 2 - 1) * jitter
    )
    fixture[:, :3] += origins
    fixture[:, 3] = 1
    env.scene["fixture"].write_mocap_pose_to_sim(fixture, env_ids=ids)
    obj = torch.zeros(n, 13, device=env.device)
    obj[:, :2] = (
        torch.tensor(SOURCE_XY, device=env.device)
        + (torch.rand(n, 2, device=env.device) * 2 - 1) * jitter
    )
    obj[:, 2] = 0.056 if cfg.task == "transfer" else 0.009
    obj[:, 3] = 1
    if cfg.task == "plug":
        term = env.action_manager.get_term("arms")
        xy = torch.tensor(plug.SPAWN_CENTER, device=env.device).expand(n, -1).clone()
        if cfg.randomize:
            xy += (torch.rand(n, 2, device=env.device) * 2 - 1) * torch.tensor(
                plug.SPAWN_HALF, device=env.device
            )
        tcp = torch.cat((xy, torch.full((n, 1), plug.SPAWN_Z, device=env.device)), dim=-1)
        tcp += torch.tensor(plug.GRIP_OFFSET, device=env.device)
        q = term.plug_ik.reset_manip(tcp)
        joints = s.home["manipulator"].expand(n, -1).clone()
        joints[:, :5] = q
        joints[:, 5] = 0.27
        env.scene["manipulator"].write_joint_state_to_sim(
            joints, torch.zeros_like(joints), env_ids=ids
        )
        joints[:, 5] = 0.24
        env.scene["manipulator"].set_joint_position_target(joints, env_ids=ids)
        term.tcp_target[ids] = tcp
        term.gimbal_target[ids] = term.plug_ik.home
        # Full-pose reset IK is accurate to numerical tolerance. The original
        # fixed grasp defines an identity plug rotation at this pose.
        fk_pos, fk_rot = term.plug_ik.manip.fk(q)
        # Convert site rotation back to gripper body rotation, then apply weld.
        model = env.sim.mj_model
        site_quat = torch.as_tensor(
            model.site("manipulator/gripperframe").quat, device=env.device, dtype=torch.float32
        )[None]
        body_rot = fk_rot @ quat_wxyz_to_mat(site_quat).transpose(-1, -2)
        site_pos = torch.as_tensor(
            model.site("manipulator/gripperframe").pos, device=env.device, dtype=torch.float32
        )
        p_rel, q_rel = grasp_relpose()
        obj[:, :3] = term.plug_ik.manip_frame.pos_to_world(fk_pos) + (
            body_rot @ (torch.as_tensor(p_rel, device=env.device, dtype=torch.float32) - site_pos)
        )
        from mjlab.utils.lab_api.math import quat_from_matrix

        obj[:, 3:7] = quat_from_matrix(
            body_rot
            @ quat_wxyz_to_mat(torch.as_tensor(q_rel, device=env.device, dtype=torch.float32)[None])
        )
        fixture[:, :2] = torch.rand(n, 2, device=env.device) * 0.05 if cfg.randomize else 0.025
        fixture[:, :2] += origins[:, :2]
        env.scene["fixture"].write_mocap_pose_to_sim(fixture, env_ids=ids)
    obj[:, :3] += origins
    env.scene["object"].write_root_state_to_sim(obj, env_ids=ids)
    s.onset[ids] = 2 + torch.rand(n, device=env.device) * 3 if cfg.occlusion == "random" else 3
    s.duration[ids] = 1 + torch.rand(n, device=env.device) * 3 if cfg.occlusion == "random" else 4
    s.side[ids] = torch.randint(0, 2, (n,), device=env.device).float() * 2 - 1
    s.hold[ids] = 0
    s.succeeded[ids] = False
    s.potential_valid[ids] = False
    env.sim.data.xfrc_applied[ids, s.object_id] = 0
    update_panel(env, torch.zeros(env.num_envs, device=env.device), ids)


def update_panel(env, t, ids=slice(None)):
    s = state(env)
    if isinstance(ids, slice):
        ids = s.env_ids[ids]
    visible = (t >= s.onset) & (t <= s.onset + s.duration)
    if s.cfg.occlusion == "clean":
        visible = torch.zeros_like(visible)
    elif s.cfg.occlusion == "static":
        visible = torch.ones_like(visible)
    pose = s.panel_pose
    pose[:, 0] = -0.045 + 0.035 * s.side
    pose[:, 1] = -0.105
    pose[:, 2] = torch.where(visible, 0.155, -1.0)
    pose[:, :3] += env.scene.env_origins
    pose[:, 3] = 1
    env.scene["occluder"].write_mocap_pose_to_sim(pose[ids], env_ids=ids)


@dataclass(kw_only=True)
class ArmsActionCfg(ActionTermCfg):
    experiment: Experiment

    def build(self, env):
        return ArmsAction(self, env)


class ArmsAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.exp = cfg.experiment
        self._raw = torch.zeros(env.num_envs, self.exp.action_dim, device=env.device)
        self.targets = {
            name: env.scene[name].data.joint_pos.clone() for name in ("manipulator", "camera_arm")
        }
        self.camera = env.scene["camera_arm"]
        self.camera_frozen = False
        self.plug_ik = PlugIK(env.sim.mj_model, env.device) if self.exp.task == "plug" else None
        self.tcp_target = torch.zeros(env.num_envs, 3, device=env.device)
        self.gimbal_target = torch.tensor(plug.GIMBAL_HOME, device=env.device).repeat(
            env.num_envs, 1
        )

    @property
    def action_dim(self):
        return self.exp.action_dim

    @property
    def raw_action(self):
        return self._raw

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        for name in self.targets:
            self.targets[name][ids] = self._env.scene[name].data.joint_pos[ids]
        if self.exp.task == "plug":
            self.targets["manipulator"][ids, 5] = 0.24

    def process_actions(self, actions):
        cfg, env = self.exp, self._env
        s = state(env)
        self._raw.copy_(actions.clamp(-1, 1))
        t = env.episode_length_buf * env.step_dt
        moving = t >= cfg.initial_seconds
        if cfg.task == "plug":
            requested = self.plug_ik.low + (self._raw[:, :3] + 1) * 0.5 * (
                self.plug_ik.high - self.plug_ik.low
            )
            self.tcp_target += (requested - self.tcp_target).clamp(
                -plug.TCP_MAX_DELTA, plug.TCP_MAX_DELTA
            ) * moving[:, None]
            q = self.plug_ik.manip_step(self.tcp_target, self.targets["manipulator"][:, :5])
            self.targets["manipulator"][:, :5] = torch.where(
                moving[:, None], q, self.targets["manipulator"][:, :5]
            )
        else:
            self.targets["manipulator"][:, :5] += (
                self._raw[:, :5] * cfg.joint_step * moving[:, None]
            )
        if cfg.task == "transfer":
            self.targets["manipulator"][:, 5] = torch.where(
                moving, (self._raw[:, 5] + 1) * 0.6, self.targets["manipulator"][:, 5]
            )
        if cfg.camera_control and not self.camera_frozen:
            enabled = torch.ones_like(moving) if cfg.condition == "active" else ~moving
            if cfg.task == "plug":
                self.gimbal_target += (
                    self._raw[:, 3:] * self.plug_ik.gimbal_delta * enabled[:, None]
                )
                self.gimbal_target.clamp_(self.plug_ik.gimbal_low, self.plug_ik.gimbal_high)
                q = self.plug_ik.camera_step(self.gimbal_target, self.targets["camera_arm"][:, :5])
                self.targets["camera_arm"][:, :5] = torch.where(
                    enabled[:, None], q, self.targets["camera_arm"][:, :5]
                )
            else:
                self.targets["camera_arm"][:, :5] += (
                    self._raw[:, cfg.manip_dim :] * cfg.joint_step * enabled[:, None]
                )
        elif cfg.condition == "scheduled":
            if cfg.task == "plug":
                alpha = (t / max(cfg.initial_seconds, 0.6)).clamp(0, 1)
                self.gimbal_target[:] = self.plug_ik.home + alpha[:, None] * torch.tensor(
                    [-0.035, -0.035, -0.04, 0.18, -0.20], device=env.device
                )
                self.targets["camera_arm"][:, :5] = self.plug_ik.camera_step(
                    self.gimbal_target, self.targets["camera_arm"][:, :5]
                )
            else:
                self._joint_camera_schedule(t, s)
        for name, target in self.targets.items():
            limits = env.scene[name].data.joint_pos_limits
            target.clamp_(limits[..., 0], limits[..., 1])
        update_panel(env, t)
        if cfg.perturb_push:
            force = torch.where((t >= 4) & (t < 4.12), 0.025 * s.side, 0.0)
            env.sim.data.xfrc_applied[:, s.object_id, 0] = force

    def _joint_camera_schedule(self, t, s):
        cfg = self.exp
        fraction = (
            (t - cfg.initial_seconds) / (cfg.episode_seconds - cfg.initial_seconds) * 2
        ).clamp(0, 2)
        idx = fraction.long().clamp(max=1)
        alpha = (fraction - idx)[:, None]
        desired = s.home["camera_path"][idx] * (1 - alpha) + s.home["camera_path"][idx + 1] * alpha
        self.targets["camera_arm"][:, :5] += (desired - self.targets["camera_arm"][:, :5]).clamp(
            -cfg.joint_step, cfg.joint_step
        )

    def apply_actions(self):
        self._entity.set_joint_position_target(self.targets["manipulator"])
        self.camera.set_joint_position_target(self.targets["camera_arm"])


def proprio(env):
    """Only measurable robot state; no object state, goal pose, reward, or occlusion timer."""
    pieces = []
    for name in ("manipulator", "camera_arm"):
        pieces += [env.scene[name].data.joint_pos, env.scene[name].data.joint_vel * 0.1]
    term = env.action_manager.get_term("arms")
    pieces += [term.targets["manipulator"], term.targets["camera_arm"]]
    if term.exp.task == "plug":
        pieces += [term.tcp_target, term.gimbal_target]
    # Pose of each image's camera is reconstructable from the joint angles. Include
    # fixed calibration explicitly so searched external placements are distinguishable.
    fixed = state(env).fixed_calibration
    pieces += [fixed.expand(env.num_envs, -1)]
    pieces += [(env.episode_length_buf * env.step_dt / term.exp.episode_seconds)[:, None]]
    return torch.cat(pieces, dim=-1)


def rgb(env, sensor):
    # uint8 until the CNN: avoids duplicating full float32 images in rollout storage.
    return env.scene[sensor].data.rgb.permute(0, 3, 1, 2)


def critic_state(env):
    tcp, obj, tip, goal = positions(env)
    origins = env.scene.env_origins
    s = state(env)
    return torch.cat(
        [
            proprio(env),
            tcp - origins,
            obj - origins,
            tip - origins,
            goal - origins,
            env.scene["object"].data.root_link_quat_w,
            env.scene["object"].data.root_link_vel_w,
            s.onset[:, None],
            s.duration[:, None],
            s.side[:, None],
        ],
        dim=-1,
    )


def instantaneous_success(env):
    tcp, obj, tip, goal = positions(env)
    cfg = state(env).cfg
    speed = torch.linalg.vector_norm(env.scene["object"].data.root_link_lin_vel_w, dim=-1)
    if cfg.task == "plug":
        return torch.linalg.vector_norm(obj - goal, dim=-1) < plug.SUCCESS_DISTANCE
    tol, ztol, withdraw = (0.018, 0.006, 0.045) if cfg.task == "transfer" else (0.015, 0.004, 0.025)
    return (
        (torch.linalg.vector_norm(obj[:, :2] - goal[:, :2], dim=-1) < tol)
        & ((obj[:, 2] - goal[:, 2]).abs() < ztol)
        & (torch.linalg.vector_norm(tcp - obj, dim=-1) > withdraw)
        & (speed < 0.035)
    )


def success(env):
    s = state(env)
    if s.last_success_step != env.common_step_counter:
        s.hold[:] = torch.where(instantaneous_success(env), s.hold + 1, 0)
        s.succeeded[:] = s.hold >= 3
        s.last_success_step = env.common_step_counter
    return s.succeeded


def failure(env):
    obj = env.scene["object"].data.root_link_pos_w - env.scene.env_origins
    q = env.scene["manipulator"].data.joint_pos
    return (obj[:, 2] < -0.035) | (obj[:, :2].abs().amax(-1) > 0.42) | ~torch.isfinite(q).all(-1)


def potential(env):
    tcp, obj, tip, goal = positions(env)
    cfg = state(env).cfg
    if cfg.task == "plug":
        xy = torch.linalg.vector_norm(obj[:, :2] - goal[:, :2], dim=-1)
        z = (obj[:, 2] - goal[:, 2]).abs()
        return torch.exp(-xy / 0.035) + torch.exp(-xy / 0.008) * torch.exp(-z / 0.025)
    reach = torch.exp(-torch.linalg.vector_norm(tcp - obj, dim=-1) / 0.05)
    place = torch.exp(-torch.linalg.vector_norm(obj - goal, dim=-1) / 0.045)
    if cfg.task == "transfer":
        lift = ((obj[:, 2] - env.scene.env_origins[:, 2] - 0.056) / 0.05).clamp(0, 1)
        return reach * 0.3 + lift * 0.3 + place * 1.4
    return reach * 0.3 + place * 1.7


def task_reward(env):
    s = state(env)
    phi = potential(env)
    # Delta shaping avoids rewarding indefinite hovering near the goal.
    progress = torch.where(s.potential_valid, phi - s.previous_potential, 0.0)
    s.previous_potential.copy_(phi)
    s.potential_valid[:] = True
    action = env.action_manager.action
    # Keep the manipulation penalty identical when camera action dimensions exist.
    motion_cost = 0.0005 * action.square().sum(-1) / s.cfg.manip_dim
    return progress * 3 + success(env).float() * 10 - 0.005 - motion_cost


def success_metric(env):
    return success(env).float()
