"""Vectorized task logic; plug IK runs once per control step, without host transfers."""

from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from . import manipulation, plug, plug_learning, pushing, transfer
from .config import ARM_JOINTS, MANIP_BASE, SOURCE_XY, TARGET_XY, Experiment
from .native import calibration
from .occlusion import dynamic_position_torch, sample_dynamic_torch
from .robots.kinematics import SO101Chain, quat_wxyz_to_mat
from .robots.plug_control import PlugIK
from .scenes import grasp_relpose, yaw_quat


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
        self.panel_center = torch.zeros(n, 3, device=device)
        self.panel_travel = torch.zeros(n, device=device)
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
        self.object_qpos_address = int(env.sim.mj_model.joint("object/free").qposadr[0])
        self.object_dof_address = int(env.sim.mj_model.joint("object/free").dofadr[0])
        self.transfer_lifted = torch.zeros(n, dtype=torch.bool, device=device)
        self.plug_raised = torch.zeros(n, dtype=torch.bool, device=device)
        self.plug_aligned = torch.zeros(n, dtype=torch.bool, device=device)
        if transfer.is_v2(cfg) or cfg.task_revision == "push_v2":
            names = ["manipulator/" + joint for joint in ARM_JOINTS]
            self.tcp_chain = SO101Chain(
                env.sim.mj_model,
                "manipulator/gripperframe",
                names,
                device=device,
                dtype=torch.float32,
            )
            self.manip_qpos_addresses = [
                int(env.sim.mj_model.joint(name).qposadr[0]) for name in names
            ]
            self.manip_base = torch.tensor(MANIP_BASE, device=device)
            if cfg.task_revision == "push_v2":
                self.push_tool_offset = pushing.tool_offset(env.sim.mj_model, device)
            self.current_tcp_fk = self.tcp_chain.fk
            if str(device).startswith("cuda"):
                self.current_tcp_fk = torch.compile(self.current_tcp_fk, fullgraph=True)
        self.env_ids = torch.arange(n, device=device)
        self.fixed_calibration = torch.tensor(
            (*cfg.fixed_position, *cfg.fixed_lookat), device=device
        )
        self.panel_pose = torch.zeros(n, 7, device=device)
        self.panel_quat = torch.tensor(
            yaw_quat(cfg.dynamic_panel_yaw or 0.0), device=device, dtype=torch.float32
        )
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
        if getattr(s.cfg, "reward_profile", None) in ("plug_staged", "plug_insertion"):
            term = env.action_manager.get_term("arms")
            tcp, _ = term.plug_ik.manip.fk(env.sim.data.qpos[:, term.plug_qpos_addresses])
            tcp = term.plug_ik.manip_frame.pos_to_world(tcp) + env.scene.env_origins
            obj = env.sim.data.qpos[:, s.object_qpos_address : s.object_qpos_address + 3]
            return tcp, obj, obj, goal
    if transfer.is_v2(s.cfg) or s.cfg.task_revision == "push_v2":
        tcp, _ = s.current_tcp_fk(env.sim.data.qpos[:, s.manip_qpos_addresses])
        tcp = tcp + s.manip_base + env.scene.env_origins
        obj = env.sim.data.qpos[:, s.object_qpos_address : s.object_qpos_address + 3]
        return tcp, obj, obj, goal
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
    jitter = cfg.reset_xy_jitter if cfg.randomize else 0.0
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
    if cfg.push_target_layout in ("arc_v1", "arc_v2"):
        samples = (
            torch.rand(n, 2, device=env.device) if cfg.randomize
            else torch.full((n, 2), 0.5, device=env.device)
        )
        fixture[:, :2] = obj[:, :2] + pushing.arc_offset(samples, cfg.push_target_layout) + origins[:, :2]
        env.scene["fixture"].write_mocap_pose_to_sim(fixture, env_ids=ids)
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
        term.wrist_yaw_target[ids] = 0
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
    if cfg.occlusion == "dynamic":
        params = sample_dynamic_torch(n, cfg, env.device)
        for name in ("onset", "duration", "side"):
            getattr(s, name)[ids] = params[name]
        s.panel_center[ids] = params["center"]
        s.panel_travel[ids] = params["travel"]
    else:
        s.onset[ids] = 2 + torch.rand(n, device=env.device) * 3 if cfg.occlusion == "random" else 3
        s.duration[ids] = (
            1 + torch.rand(n, device=env.device) * 3 if cfg.occlusion == "random" else 4
        )
        s.side[ids] = torch.randint(0, 2, (n,), device=env.device).float() * 2 - 1
    s.hold[ids] = 0
    s.transfer_lifted[ids] = False
    s.plug_raised[ids] = False
    s.plug_aligned[ids] = False
    s.succeeded[ids] = False
    s.potential_valid[ids] = False
    if cfg.manip_control == "cartesian":
        env.action_manager.get_term("arms").tcp_target[ids] = s.home["tcp"]
    env.sim.data.xfrc_applied[ids, s.object_id] = 0
    update_panel(env, torch.zeros(env.num_envs, device=env.device), ids)


def update_panel(env, t, ids=slice(None)):
    s = state(env)
    if isinstance(ids, slice):
        ids = s.env_ids[ids]
    if s.cfg.occlusion == "dynamic":
        position = dynamic_position_torch(
            t[ids],
            s.onset[ids],
            s.duration[ids],
            s.side[ids],
            s.panel_center[ids],
            s.panel_travel[ids],
        )
    else:
        visible = (t[ids] >= s.onset[ids]) & (t[ids] <= s.onset[ids] + s.duration[ids])
        if s.cfg.occlusion == "clean":
            visible = torch.zeros_like(visible)
        elif s.cfg.occlusion == "static":
            visible = torch.ones_like(visible)
        position = torch.stack(
            (
                -0.045 + 0.035 * s.side[ids],
                torch.full_like(t[ids], -0.105),
                torch.where(visible, 0.155, -1.0),
            ),
            dim=-1,
        )
    # Subset reset changes only selected environments, including this cached pose.
    s.panel_pose[ids, :3] = position + env.scene.env_origins[ids]
    s.panel_pose[ids, 3:] = s.panel_quat
    env.scene["occluder"].write_mocap_pose_to_sim(s.panel_pose[ids], env_ids=ids)


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
        if self.plug_ik is not None:
            self.plug_qpos_addresses = [
                int(env.sim.mj_model.joint(name).qposadr[0])
                for name in self.plug_ik.manip.joint_names
            ]
        self.tcp_target = torch.zeros(env.num_envs, 3, device=env.device)
        self.cartesian = (
            manipulation.CartesianIK(env.sim.mj_model, self.exp.task, env.device)
            if self.exp.manip_control == "cartesian"
            else None
        )
        self.wrist_yaw_target = torch.zeros(env.num_envs, device=env.device)
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
            if cfg.plug_control == "delta5mm":
                requested = (self.tcp_target + plug_learning.STEP_METERS * self._raw[:, :3]).clamp(
                    self.plug_ik.low, self.plug_ik.high
                )
            else:
                requested = self.plug_ik.low + (self._raw[:, :3] + 1) * 0.5 * (
                    self.plug_ik.high - self.plug_ik.low
                )
            self.tcp_target += (requested - self.tcp_target).clamp(
                -plug.TCP_MAX_DELTA, plug.TCP_MAX_DELTA
            ) * moving[:, None]
            if cfg.plug_wrist_yaw:
                desired_yaw = self._raw[:, 3] * plug.WRIST_YAW_LIMIT
                self.wrist_yaw_target += (desired_yaw - self.wrist_yaw_target).clamp(
                    -plug.WRIST_YAW_MAX_DELTA, plug.WRIST_YAW_MAX_DELTA
                ) * moving
            q = self.plug_ik.manip_step(
                self.tcp_target,
                self.targets["manipulator"][:, :5],
                self.wrist_yaw_target if cfg.plug_wrist_yaw else None,
            )
            self.targets["manipulator"][:, :5] = torch.where(
                moving[:, None], q, self.targets["manipulator"][:, :5]
            )
        elif self.cartesian is not None:
            # Absolute workspace actions, bounded Cartesian slew, constant hand orientation.
            requested = self.cartesian.low + (self._raw[:, :3] + 1) * 0.5 * (
                self.cartesian.high - self.cartesian.low
            )
            self.tcp_target += (requested - self.tcp_target).clamp(
                -self.cartesian.delta, self.cartesian.delta
            ) * moving[:, None]
            q = self.cartesian.solve(self.tcp_target, self.targets["manipulator"][:, :5])
            self.targets["manipulator"][:, :5] = torch.where(
                moving[:, None], q, self.targets["manipulator"][:, :5]
            )
        else:
            self.targets["manipulator"][:, :5] += (
                self._raw[:, :5] * cfg.joint_step * moving[:, None]
            )
        if cfg.task == "transfer":
            self.targets["manipulator"][:, 5] = torch.where(
                moving,
                (self._raw[:, cfg.manip_dim - 1] + 1) * 0.6,
                self.targets["manipulator"][:, 5],
            )
        if cfg.camera_control and not self.camera_frozen:
            enabled = torch.ones_like(moving) if cfg.condition == "active" else ~moving
            if cfg.task == "plug":
                self.gimbal_target += (
                    self._raw[:, cfg.manip_dim :] * self.plug_ik.gimbal_delta * enabled[:, None]
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
    if term.exp.manip_control == "cartesian":
        pieces += [term.tcp_target]
    if term.exp.task == "plug":
        pieces += [term.tcp_target, term.gimbal_target]
        if term.exp.plug_wrist_yaw:
            pieces += [term.wrist_yaw_target[:, None]]
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
    pieces = [
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
    ]
    if s.cfg.occlusion == "dynamic":
        # These are never added to proprio or camera observations. Keeping the
        # legacy critic shape lets old checkpoints load without reinterpretation.
        pieces += [s.panel_center, s.panel_travel[:, None]]
    if transfer.is_v2(s.cfg):
        pieces += [s.transfer_lifted[:, None].float()]
    if getattr(s.cfg, "reward_profile", None) == "plug_staged":
        pieces += [s.plug_raised[:, None].float(), s.plug_aligned[:, None].float()]
    return torch.cat(pieces, dim=-1)


def instantaneous_success(env):
    tcp, obj, tip, goal = positions(env)
    cfg = state(env).cfg
    if transfer.is_v2(cfg):
        return transfer.warp_features(env, state(env), tcp, obj, goal)["instantaneous_success"]
    speed = torch.linalg.vector_norm(env.scene["object"].data.root_link_lin_vel_w, dim=-1)
    if cfg.task_revision == "push_v2":
        address = state(env).object_dof_address
        speed = env.sim.data.qvel[:, address : address + 3].norm(dim=-1)
    if cfg.task == "plug":
        if cfg.success_state_sample == "current_qpos":
            address = state(env).object_qpos_address
            obj = env.sim.data.qpos[:, address : address + 3]
        aligned = torch.linalg.vector_norm(obj - goal, dim=-1) < plug.SUCCESS_DISTANCE
        if cfg.plug_wrist_yaw:
            aligned &= plug_orientation_error(env) < plug.SUCCESS_ANGLE
        return aligned
    tol, ztol, withdraw = (0.018, 0.006, 0.045) if cfg.task == "transfer" else (0.015, 0.004, 0.025)
    if cfg.task_revision == "push_v2":
        tcp = pushing.tool_position(env, state(env))
        tol = cfg.push_success_tolerance
    return (
        (torch.linalg.vector_norm(obj[:, :2] - goal[:, :2], dim=-1) < tol)
        & ((obj[:, 2] - goal[:, 2]).abs() < ztol)
        & (torch.linalg.vector_norm(tcp - obj, dim=-1) > withdraw)
        & (speed < 0.035)
        & (pushing.warp_clear(env) if cfg.task_revision == "push_v2" else True)
    )


def privileged_actor_state(env):
    """Explicit full-state diagnostic only; never added to an RGB actor's groups."""
    s = state(env)
    tcp, obj, _, goal = positions(env)
    address, dof = s.object_qpos_address, s.object_dof_address
    pieces = [
        proprio(env),
        tcp - env.scene.env_origins,
        obj - env.scene.env_origins,
        goal - env.scene.env_origins,
        obj - tcp,
        goal - obj,
        env.sim.data.qpos[:, address + 3 : address + 7],
        env.sim.data.qvel[:, dof : dof + 6],
        s.hold[:, None].float(),
    ]
    if transfer.is_v2(s.cfg):
        f = transfer.warp_features(env, s, tcp, obj, goal, update=False)
        pieces += [
            torch.stack(
                [
                    f[name].float()
                    for name in ("bilateral", "lift", "lifted", "placed", "released", "settled")
                ],
                dim=-1,
            )
        ]
    elif s.cfg.task_revision == "push_v2":
        pieces += [pushing.tool_position(env, s) - obj, pushing.warp_clear(env)[:, None].float()]
    return torch.cat(pieces, dim=-1)


def diagnostic_metric(env, name):
    """State-based learning diagnostics, sampled before reset without changing physics."""
    s = state(env)
    tcp, obj, _, goal = positions(env)
    if name == "goal_distance":
        return (obj - goal).norm(dim=-1)
    if name == "hand_distance":
        return (tcp - obj).norm(dim=-1)
    if s.cfg.task == "transfer":
        return transfer.warp_features(env, s, tcp, obj, goal, update=False)[name].float()
    if name == "at_goal":
        return ((obj[:, :2] - goal[:, :2]).norm(dim=-1) < s.cfg.push_success_tolerance).float()
    if name == "tool_clear":
        return ((pushing.tool_position(env, s) - obj).norm(dim=-1) > 0.025).float()
    raise ValueError(name)


def success(env):
    s = state(env)
    if s.last_success_step != env.common_step_counter:
        s.hold[:] = torch.where(instantaneous_success(env), s.hold + 1, 0)
        s.succeeded[:] = s.hold >= (transfer.HOLD_STEPS if transfer.is_v2(s.cfg) else 3)
        s.last_success_step = env.common_step_counter
    return s.succeeded


def failure(env):
    s = state(env)
    obj = (
        env.sim.data.qpos[:, s.object_qpos_address : s.object_qpos_address + 3]
        if transfer.is_v2(s.cfg) or s.cfg.task_revision == "push_v2"
        else env.scene["object"].data.root_link_pos_w
    ) - env.scene.env_origins
    q = env.scene["manipulator"].data.joint_pos
    return (obj[:, 2] < -0.035) | (obj[:, :2].abs().amax(-1) > 0.42) | ~torch.isfinite(q).all(-1)


def plug_orientation_error(env):
    """Physical object angle from identity, invariant to quaternion sign/scale."""
    address = state(env).object_qpos_address
    quat = env.sim.data.qpos[:, address + 3 : address + 7]
    return 2 * torch.atan2(torch.linalg.vector_norm(quat[:, 1:], dim=-1), quat[:, 0].abs())


def potential(env):
    tcp, obj, tip, goal = positions(env)
    cfg = state(env).cfg
    if cfg.reward_profile == "transfer_staged":
        return transfer.potential(transfer.warp_features(env, state(env), tcp, obj, goal))
    if cfg.task == "plug":
        xy = torch.linalg.vector_norm(obj[:, :2] - goal[:, :2], dim=-1)
        z = (obj[:, 2] - goal[:, 2]).abs()
        precision = torch.exp(-xy / 0.008) * torch.exp(-z / 0.025)
        if cfg.plug_wrist_yaw:
            precision *= torch.exp(-plug_orientation_error(env) / 0.2)
        return torch.exp(-xy / 0.035) + precision
    reach = torch.exp(-torch.linalg.vector_norm(tcp - obj, dim=-1) / 0.05)
    place = torch.exp(-torch.linalg.vector_norm(obj - goal, dim=-1) / 0.045)
    if cfg.task == "transfer":
        lift = ((obj[:, 2] - env.scene.env_origins[:, 2] - 0.056) / 0.05).clamp(0, 1)
        return reach * 0.3 + lift * 0.3 + place * 1.4
    return reach * 0.3 + place * 1.7


def task_reward(env):
    s = state(env)
    action = env.action_manager.action
    if s.cfg.reward_profile == "plug_insertion":
        _, _, _, goal = positions(env)
        address = s.object_qpos_address
        pose = env.sim.data.qpos[:, address : address + 7]
        return plug_learning.insertion_reward(
            pose[:, :3], goal, pose[:, 3:7] if s.cfg.plug_wrist_yaw else None
        )
    if s.cfg.reward_profile == "plug_staged":
        tcp, obj, _, goal = positions(env)
        phi = plug_learning.features(env, s, tcp, obj, goal)
        effort = 0.001 * action[:, : s.cfg.manip_dim].square().mean(dim=-1)
        return 0.2 * phi + plug_learning.SUCCESS_BONUS * success(env).float() - 0.005 - effort
    if s.cfg.reward_profile in ("transfer_dense", "transfer_transport", "push_dense"):
        tcp, obj, _, goal = positions(env)
        if s.cfg.task == "transfer":
            f = transfer.warp_features(env, s, tcp, obj, goal)
            density = (
                manipulation.transfer_transport_density
                if s.cfg.reward_profile == "transfer_transport"
                else manipulation.transfer_density
            )
            phi = density(f, tcp, obj)
        else:
            phi = manipulation.push_density(tcp, obj, goal, s.cfg.push_success_tolerance)
        effort = 0.0005 * action[:, : s.cfg.manip_dim].square().mean(dim=-1)
        return 0.2 * phi + 100 * success(env).float() - 0.005 - effort
    if s.cfg.reward_profile == "transfer_staged":
        phi = potential(env)
        # Absorbing terminal potential prevents a timeout or a repeated cycle
        # from collecting the value of an unfinished grasp/lift/hover.
        terminal = env.termination_manager.dones
        next_phi = torch.where(terminal, torch.zeros_like(phi), phi)
        shaping = torch.where(
            s.potential_valid, transfer.DISCOUNT * next_phi - s.previous_potential, 0.0
        )
        s.previous_potential.copy_(next_phi)
        s.potential_valid[:] = True
        # Camera-control dimensionality must not change manipulation effort cost.
        effort = 0.0005 * action[:, : s.cfg.manip_dim].square().mean(dim=-1)
        return 3 * shaping + transfer.SUCCESS_BONUS * success(env).float() - 0.005 - effort
    # Preserve the same time/action costs across reward profiles and conditions.
    motion_cost = 0.0005 * action.square().sum(-1) / s.cfg.manip_dim
    if s.cfg.reward_profile == "legacy_log_hold":
        _, _, _, goal = positions(env)
        address = s.object_qpos_address
        obj = env.sim.data.qpos[:, address : address + 3]
        distance = torch.linalg.vector_norm(obj - goal, dim=-1)
        if s.cfg.plug_wrist_yaw:
            # Convert rotational misalignment to a physical displacement scale
            # at a 30 mm lever arm; translation-only historical profiles unchanged.
            distance = torch.sqrt(distance.square() + (0.03 * plug_orientation_error(env)).square())
        shaping = -torch.log1p(100 * distance)
        return shaping + 1000 * success(env).float() - 0.005 - motion_cost
    phi = potential(env)
    # Delta shaping avoids rewarding indefinite hovering near the goal.
    progress = torch.where(s.potential_valid, phi - s.previous_potential, 0.0)
    s.previous_potential.copy_(phi)
    s.potential_valid[:] = True
    return progress * 3 + success(env).float() * 10 - 0.005 - motion_cost


def success_metric(env):
    return success(env).float()
