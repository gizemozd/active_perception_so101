"""Native MuJoCo reference runner for physics and camera diagnostics (no learning)."""

from functools import lru_cache

import mujoco
import numpy as np
import torch

from .config import (
    ARM_JOINTS,
    CAMERA_BASE,
    CAMERA_BASE_YAW,
    CAMERA_HOME,
    CAMERA_LOOKAT,
    JOINTS,
    MANIP_BASE,
    SOURCE_XY,
    STOW,
    TARGET_XY,
    Experiment,
)
from .robots.kinematics import BaseFrame, SO101Chain, grasp_site_rotation
from .scenes import grasp_relpose, native_spec


class Kinematics:
    def __init__(self, model):
        self.model = model
        self.chains = {}
        self.frames = {}
        for name, site, base, yaw in (
            ("manipulator", "gripperframe", MANIP_BASE, 0),
            ("camera_arm", "wrist_cam_site", CAMERA_BASE, CAMERA_BASE_YAW),
        ):
            self.chains[name] = SO101Chain(
                model, name + "/" + site, [name + "/" + j for j in ARM_JOINTS], dtype=torch.float64
            )
            self.frames[name] = BaseFrame(base, yaw)
        self.rotation = torch.as_tensor(
            grasp_site_rotation(model, prefix="manipulator/", yaw=-np.pi / 2)
        )

    def manip(self, target, seed=None):
        frame, chain = self.frames["manipulator"], self.chains["manipulator"]
        p = frame.pos_to_base(torch.as_tensor(target, dtype=torch.float64).reshape(1, 3))
        r = frame.mat_to_base(self.rotation).unsqueeze(0)
        if seed is None:
            q = chain.solve_ik_staged(
                p, r, site_axis=(1, 0, 0), iters_axis=80, iters_full=50, damping=0.008
            )
        else:
            q = chain.solve_ik(
                p,
                torch.as_tensor(seed, dtype=torch.float64).reshape(1, 5),
                target_rot=r,
                iters=12,
                damping=0.006,
            )
        return q[0].numpy()

    def camera(self, position, lookat=CAMERA_LOOKAT, seed=None):
        frame, chain = self.frames["camera_arm"], self.chains["camera_arm"]
        p = frame.pos_to_base(torch.as_tensor(position, dtype=torch.float64).reshape(1, 3))
        axis = torch.as_tensor(np.asarray(lookat) - position, dtype=torch.float64).reshape(1, 3)
        axis /= torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        seed = STOW[:5] if seed is None else seed
        q = chain.solve_ik(
            p,
            torch.as_tensor(seed, dtype=torch.float64).reshape(1, 5),
            target_axis=frame.vec_to_base(axis),
            iters=100,
            damping=0.008,
        )
        return q[0].numpy()


@lru_cache(maxsize=12)
def calibration(task):
    """CPU IK once at construction; the training hot path uses joint targets."""
    model = native_spec(Experiment(task=task)).compile()
    ik = Kinematics(model)
    if task == "plug":
        tcp = (-0.025, -0.0015, 0.102)
    elif task == "transfer":
        tcp = (*SOURCE_XY, 0.11)
    else:
        tcp = (-0.025, -0.105, 0.040)
    manip = np.r_[ik.manip(tcp), 0.24 if task == "plug" else (0.9 if task == "transfer" else 0.0)]
    cam = np.r_[ik.camera(np.array(CAMERA_HOME)), 0.6]
    path = [cam[:5]]
    for p, target in (
        (np.array(CAMERA_HOME) + [-0.035, 0, 0.01], (*SOURCE_XY, 0.04)),
        (np.array(CAMERA_HOME) + [0.035, -0.015, 0.01], (*TARGET_XY, 0.035)),
    ):
        path.append(ik.camera(p, target, cam[:5]))
    data = mujoco.MjData(model)
    for arm, q in (("manipulator", manip), ("camera_arm", cam)):
        for joint, value in zip(JOINTS, q, strict=True):
            data.qpos[model.joint(arm + "/" + joint).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    body = model.body("manipulator/gripper").id
    p_rel, q_rel = grasp_relpose()
    pos = data.xpos[body] + data.xmat[body].reshape(3, 3) @ p_rel
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, data.xquat[body], q_rel)
    return {
        "manipulator": manip,
        "camera_arm": cam,
        "camera_path": np.array(path),
        "plug_pose": np.r_[pos, quat],
    }


def schedule_fraction(time, cfg):
    return np.clip(
        (time - cfg.initial_seconds) / max(cfg.episode_seconds - cfg.initial_seconds, 1e-6) * 2,
        0,
        2,
    )


def occluder_position(time, onset, duration, side, cfg):
    if cfg.occlusion == "clean":
        return np.array([0.0, 0.0, -1.0])
    visible = cfg.occlusion == "static" or onset <= time <= onset + duration
    # Remains outside the object's motion corridor; moves around the camera sightline.
    return np.array([-0.045 + 0.035 * side, -0.105, 0.155 if visible else -1.0])


class NativeEnv:
    def __init__(self, cfg: Experiment):
        self.cfg = cfg
        self.model = native_spec(cfg).compile()
        self.data = mujoco.MjData(self.model)
        self.ik = Kinematics(self.model)
        self.home = calibration(cfg.task)
        self.qadr = {
            arm: np.array([self.model.joint(arm + "/" + j).qposadr[0] for j in JOINTS])
            for arm in ("manipulator", "camera_arm")
        }
        self.ctrlids = {
            arm: np.array([self.model.actuator(arm + "/" + j).id for j in JOINTS])
            for arm in self.qadr
        }
        self.limits = {
            arm: np.array([self.model.joint(arm + "/" + j).range for j in JOINTS])
            for arm in self.qadr
        }
        self.object_adr = self.model.joint("object/free").qposadr[0]
        self.fixture_id = self.model.body("fixture/fixture").mocapid[0]
        self.panel_id = self.model.body("occluder/panel").mocapid[0]
        self.step_count = 0
        self.reset(cfg.seed)

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.targets = {name: self.home[name].copy() for name in self.qadr}
        for name, q in self.targets.items():
            self.data.qpos[self.qadr[name]] = q
            self.data.ctrl[self.ctrlids[name]] = q
        jitter = 0.008 if self.cfg.randomize else 0.0
        goal_xy = np.array(TARGET_XY if self.cfg.task != "push" else (-0.025, 0.055))
        self.data.mocap_pos[self.fixture_id] = np.r_[
            goal_xy + self.rng.uniform(-jitter, jitter, 2), 0.0
        ]
        pose = np.array([*SOURCE_XY, 0.056 if self.cfg.task == "transfer" else 0.009, 1, 0, 0, 0])
        pose[:2] += self.rng.uniform(-jitter, jitter, 2)
        if self.cfg.task == "plug":
            pose = self.home["plug_pose"].copy()
        self.data.qpos[self.object_adr : self.object_adr + 7] = pose
        self.onset = float(self.rng.uniform(2, 5)) if self.cfg.occlusion == "random" else 3.0
        self.duration = float(self.rng.uniform(1, 4)) if self.cfg.occlusion == "random" else 4.0
        self.side = float(self.rng.choice([-1, 1]))
        self.data.mocap_pos[self.panel_id] = occluder_position(
            0, self.onset, self.duration, self.side, self.cfg
        )
        mujoco.mj_forward(self.model, self.data)

    @property
    def tcp(self):
        return self.data.site("manipulator/gripperframe").xpos.copy()

    @property
    def object_pos(self):
        return self.data.xpos[self.model.body("object/object").id].copy()

    @property
    def goal(self):
        return self.data.site("fixture/goal").xpos.copy()

    def step(self, action):
        cfg = self.cfg
        action = np.asarray(action, dtype=float)
        if action.shape != (cfg.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Action must have the configured shape and finite values")
        action = np.clip(action, -1, 1)
        t = self.step_count * cfg.step_dt
        if t >= cfg.initial_seconds:
            self.targets["manipulator"][:5] += cfg.joint_step * action[:5]
            if cfg.task == "transfer":
                self.targets["manipulator"][5] = (action[5] + 1) * 0.6
        if cfg.camera_control and (cfg.condition == "active" or t < cfg.initial_seconds):
            self.targets["camera_arm"][:5] += cfg.joint_step * action[cfg.manip_dim :]
        elif cfg.condition == "scheduled":
            fraction = schedule_fraction(t, cfg)
            idx = min(int(fraction), 1)
            desired = (1 - (fraction - idx)) * self.home["camera_path"][idx] + (
                fraction - idx
            ) * self.home["camera_path"][idx + 1]
            self.targets["camera_arm"][:5] += np.clip(
                desired - self.targets["camera_arm"][:5], -cfg.joint_step, cfg.joint_step
            )
        for arm in self.targets:
            self.targets[arm] = np.clip(
                self.targets[arm], self.limits[arm][:, 0], self.limits[arm][:, 1]
            )
            self.data.ctrl[self.ctrlids[arm]] = self.targets[arm]
        self.data.mocap_pos[self.panel_id] = occluder_position(
            t, self.onset, self.duration, self.side, cfg
        )
        if cfg.perturb_push and 4.0 <= t < 4.12:
            self.data.xfrc_applied[self.model.body("object/object").id, 0] = 0.025 * self.side
        else:
            self.data.xfrc_applied[:] = 0
        mujoco.mj_step(self.model, self.data, nstep=cfg.decimation)
        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1
        return self.success()

    def success(self):
        obj, goal = self.object_pos, self.goal
        speed = np.linalg.norm(self.data.qvel[-6:-3])
        if self.cfg.task == "plug":
            tip = self.data.site("object/tip").xpos
            quat = self.data.xquat[self.model.body("object/object").id]
            return bool(
                np.linalg.norm(tip[:2] - goal[:2]) < self.cfg.clearance * 0.8
                and abs(tip[2] - goal[2]) < 0.003
                and abs(quat[0]) > np.cos(np.deg2rad(5) / 2)
                and speed < 0.05
            )
        if self.cfg.task == "transfer":
            return bool(
                np.linalg.norm(obj[:2] - goal[:2]) < 0.018
                and abs(obj[2] - goal[2]) < 0.006
                and np.linalg.norm(self.tcp - obj) > 0.045
                and speed < 0.035
            )
        return bool(
            np.linalg.norm(obj[:2] - goal[:2]) < 0.015
            and abs(obj[2] - goal[2]) < 0.004
            and np.linalg.norm(self.tcp - obj) > 0.025
            and speed < 0.035
        )


class ScriptedPolicy:
    """Privileged diagnostic controller; never a policy-training observation source."""

    def __init__(self, env: NativeEnv):
        self.env = env
        self.stage = 0
        self.ticks = 0
        self.qseed = env.targets["manipulator"][:5].copy()
        self.camera_q = env.home["camera_arm"][:5].copy()
        self.start_object = env.object_pos.copy()
        self.retreat = None

    def __call__(self):
        env, cfg = self.env, self.env.cfg
        obj, goal = env.object_pos, env.goal
        opened = -1.0
        if cfg.task == "plug":
            # Move above the socket, then lower the pin through its physical opening.
            p = goal + np.array([0, -0.0015, 0.055])
            if self.stage == 0:
                p[2] += 0.045
        elif cfg.task == "transfer":
            p = self.start_object.copy()
            if self.stage == 0:
                p[2] += 0.055
                opened = 0.5
            elif self.stage == 1:
                p[2] += 0.003
                opened = 0.5
            elif self.stage == 2:
                p[2] += 0.003
            elif self.stage == 3:
                p = obj.copy()
                p[2] = 0.110
            elif self.stage == 4:
                p = goal.copy()
                p[0] -= 0.035
                p[2] = 0.110
            elif self.stage == 5:
                p = goal.copy()
                p[2] += 0.004
            else:
                p = goal.copy()
                opened = 0.65
                if self.stage >= 7:
                    p[0] -= 0.065
                    p[2] += 0.055
            p[1] -= 0.008
        else:
            direction = goal[:2] - obj[:2]
            distance = np.linalg.norm(direction)
            direction /= max(distance, 1e-6)
            p = obj.copy()
            # Contact with the 8 mm radius tip at the rear of the 24 mm block.
            p[:2] -= direction * (0.028 if self.stage == 0 else 0.017)
            p[2] = 0.040
            if distance < 0.009 or self.stage >= 2:
                self.stage = 2
                if self.retreat is None:
                    self.retreat = env.tcp + np.array([0, 0, 0.065])
                p = self.retreat
        self.qseed = env.ik.manip(p, self.qseed)
        action = np.zeros(cfg.action_dim)
        action[:5] = np.clip((self.qseed - env.targets["manipulator"][:5]) / cfg.joint_step, -1, 1)
        if cfg.task == "push":
            action[:5] = np.clip(action[:5], -0.25, 0.25)
        if cfg.task == "transfer":
            action[5] = opened
        if cfg.camera_control:
            if self.ticks % 15 == 0:
                camera_pos = np.array(CAMERA_HOME) + [0.035 * np.sin(env.data.time), 0, 0]
                self.camera_q = env.ik.camera(camera_pos, (obj + goal) / 2, self.camera_q)
            action[cfg.manip_dim :] = np.clip(
                (self.camera_q - env.targets["camera_arm"][:5]) / cfg.joint_step, -1, 1
            )
        if env.data.time >= cfg.initial_seconds:
            self.ticks += 1
            close = np.linalg.norm(env.tcp - p) < 0.006
            dwell = 22 if cfg.task == "transfer" and self.stage in (2, 6) else 10
            if (close or (cfg.task == "transfer" and self.stage in (2, 6))) and self.ticks > dwell:
                if cfg.task == "plug":
                    self.stage = 1
                elif cfg.task == "transfer":
                    self.stage = min(self.stage + 1, 7)
                elif self.stage == 0:
                    self.stage = 1
                self.ticks = 0
        return action
