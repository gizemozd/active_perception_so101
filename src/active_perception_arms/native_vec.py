"""Threaded native-MuJoCo batch of plug worlds for CPU learnability diagnostics.

This is not a visual-policy backend: nothing is rendered and the camera arm stays
parked. It reproduces the MjLab plug reset distributions, ``pose_v1`` delta
controls, current-qpos success sampling and both v1/v2 rules, so a state-based
learner can separate control/reward learnability from perception on a laptop.
MuJoCo releases the GIL inside ``mj_step``; worlds are stepped by a thread pool.

Differences from the Warp backend: float64 native physics, independent RNG, and
one compiled model per hidden-prong variant (Warp batches the meshes instead).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import mujoco
import numpy as np
import torch

from . import plug, plug_v2
from .config import JOINTS, Experiment
from .native import calibration
from .robots.plug_control import PlugIK
from .scenes import grasp_relpose, native_spec

OBSERVATIONS = ("privileged", "socket_only", "blind")


class NativePlugBatch:
    """RSL-RL compatible vector environment (``get_observations``/``step``)."""

    def __init__(self, cfg: Experiment, num_envs, *, threads=8, seed=0, observation="privileged"):
        if cfg.task != "plug" or cfg.control_revision != "pose_v1":
            raise ValueError("Native batch diagnostics cover the pose_v1 plug task")
        if observation not in OBSERVATIONS:
            raise ValueError(f"observation must be one of {OBSERVATIONS}")
        self.cfg, self.num_envs, self.observation = cfg, num_envs, observation
        self.device = "cpu"
        self.rng = np.random.default_rng(seed)
        self.variant_index = np.asarray(plug.variant_assignment(num_envs, cfg.plug_variant))
        self.models = {
            v: native_spec(replace(cfg, plug_variant=v)).compile() for v in plug.VARIANTS
        }
        self.worlds = [
            (self.models[plug.VARIANTS[i]], mujoco.MjData(self.models[plug.VARIANTS[i]]))
            for i in self.variant_index
        ]
        model = self.models["xm"]
        self.ik = PlugIK(model, dtype=torch.float64)
        self.home = calibration("plug", camera_start=cfg.camera_start)
        self.qadr = {
            arm: np.array([model.joint(f"{arm}/{j}").qposadr[0] for j in JOINTS])
            for arm in ("manipulator", "camera_arm")
        }
        self.vadr = {
            arm: np.array([model.joint(f"{arm}/{j}").dofadr[0] for j in JOINTS])
            for arm in ("manipulator", "camera_arm")
        }
        self.ctrl = {
            arm: np.array([model.actuator(f"{arm}/{j}").id for j in JOINTS])
            for arm in ("manipulator", "camera_arm")
        }
        self.limits = np.array([model.joint(f"manipulator/{j}").range for j in JOINTS])
        self.object_adr = int(model.joint("object/free").qposadr[0])
        self.object_dof = int(model.joint("object/free").dofadr[0])
        self.fixture = int(model.body("fixture/fixture").mocapid[0])
        self.gripper_body = model.body("manipulator/gripper").id
        self.p_rel, self.q_rel = grasp_relpose()
        self.offsets = np.array([plug.OFFSETS[plug.VARIANTS[i]] for i in self.variant_index])
        self.pool = ThreadPoolExecutor(threads)
        self.chunks = np.array_split(np.arange(num_envs), threads)
        n = num_envs
        self.max_episode_length = int(np.ceil(cfg.episode_seconds / cfg.step_dt - 1e-9))
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.tcp_target = np.zeros((n, 3))
        self.manip_target = np.zeros((n, 6))
        self.socket = np.zeros((n, 3))
        self.hold = np.zeros(n, dtype=int)
        self.phi = np.zeros(n)
        self.phi_valid = np.zeros(n, dtype=bool)
        self.qpos = np.zeros((n, model.nq))
        self.qvel = np.zeros((n, model.nv))
        self.num_actions = cfg.manip_dim
        self.reset(np.arange(n))
        self.obs = self._observe()

    # -- RSL-RL interface ------------------------------------------------------
    def get_observations(self):
        return self.obs

    def close(self):
        self.pool.shutdown()

    # -- state ------------------------------------------------------------------
    def _sync(self, ids=None):
        for i in range(self.num_envs) if ids is None else ids:
            _, data = self.worlds[i]
            self.qpos[i] = data.qpos
            self.qvel[i] = data.qvel

    def goal(self):
        goal = self.socket.copy()
        goal[:, :2] -= self.offsets
        goal[:, 2] = plug.INSERT_Z
        return goal

    def reset(self, ids):
        ids = np.asarray(ids)
        if len(ids) == 0:
            return
        n = len(ids)
        scale = self.cfg.plug_reset_scale if self.cfg.randomize else 0.0
        xy = (
            np.array(plug.SPAWN_CENTER)
            + self.rng.uniform(-1, 1, (n, 2)) * np.array(plug.SPAWN_HALF) * scale
        )
        socket_center = np.array((0.025, 0.025))
        socket = socket_center + self.rng.uniform(-0.025, 0.025, (n, 2)) * scale
        tcp = np.c_[xy, np.full(n, plug.SPAWN_Z)] + np.array(plug.GRIP_OFFSET)
        assist = self.rng.random(n) < self.cfg.plug_assist_fraction
        if assist.any():
            goal = socket[assist] - self.offsets[ids[assist]]
            goal += self.rng.uniform(-1, 1, goal.shape) * plug_v2.ASSIST_XY
            height = self.rng.uniform(*plug_v2.ASSIST_HEIGHT, assist.sum())
            tcp[assist] = np.c_[goal, plug.INSERT_Z + height] + np.array(plug.GRIP_OFFSET)
        q = self.ik.reset_manip(torch.as_tensor(tcp)).numpy()
        for k, i in enumerate(ids):
            model, data = self.worlds[i]
            mujoco.mj_resetData(model, data)
            joints = self.home["manipulator"].copy()
            joints[:5] = q[k]
            joints[5] = 0.27
            data.qpos[self.qadr["manipulator"]] = joints
            data.qpos[self.qadr["camera_arm"]] = self.home["camera_arm"]
            joints[5] = 0.24
            data.ctrl[self.ctrl["manipulator"]] = joints
            data.ctrl[self.ctrl["camera_arm"]] = self.home["camera_arm"]
            self.manip_target[i] = joints
            data.mocap_pos[self.fixture] = (*socket[k], 0.0)
            mujoco.mj_kinematics(model, data)
            rot = data.xmat[self.gripper_body].reshape(3, 3)
            quat = np.empty(4)
            mujoco.mju_mulQuat(quat, data.xquat[self.gripper_body], self.q_rel)
            pose = np.r_[data.xpos[self.gripper_body] + rot @ self.p_rel, quat]
            data.qpos[self.object_adr : self.object_adr + 7] = pose
            mujoco.mj_forward(model, data)
            self.socket[i] = (*socket[k], 0.0)
        self.tcp_target[ids] = tcp
        self.hold[ids] = 0
        self.phi_valid[ids] = False
        self.episode_length_buf[ids] = 0
        self._sync(ids)

    def object_pose(self):
        return self.qpos[:, self.object_adr : self.object_adr + 7]

    def tcp(self):
        q = torch.as_tensor(self.qpos[:, self.qadr["manipulator"][:5]])
        pos, _ = self.ik.manip.fk(q)
        return self.ik.manip_frame.pos_to_world(pos).numpy()

    def _observe(self):
        cfg, n = self.cfg, self.num_envs
        obj = self.object_pose()
        goal = self.goal()
        tcp = self.tcp()
        t = self.episode_length_buf.numpy() * cfg.step_dt
        pieces = [
            self.qpos[:, self.qadr["manipulator"]],
            0.1 * self.qvel[:, self.vadr["manipulator"]],
            self.manip_target,
            self.tcp_target,
            tcp,
            obj[:, :3],
            obj[:, 3:],
            self.qvel[:, self.object_dof : self.object_dof + 3],
            0.1 * self.qvel[:, self.object_dof + 3 : self.object_dof + 6],
            (t / cfg.episode_seconds)[:, None],
            (t >= cfg.initial_seconds)[:, None].astype(float),
            np.minimum(self.hold, 3)[:, None] / 3,
        ]
        if self.observation == "privileged":
            pieces += [goal, goal - obj[:, :3], goal - tcp]
        elif self.observation == "socket_only":
            pieces += [self.socket, self.socket - obj[:, :3]]
        state = torch.as_tensor(np.concatenate(pieces, axis=-1), dtype=torch.float32)
        critic = torch.as_tensor(
            np.concatenate([state.numpy(), goal, goal - obj[:, :3], self.offsets], axis=-1),
            dtype=torch.float32,
        )
        from tensordict import TensorDict

        return TensorDict({"policy": state, "critic": critic}, batch_size=[n])

    # -- dynamics -----------------------------------------------------------------
    def _physics(self, chunk):
        for i in chunk:
            model, data = self.worlds[i]
            mujoco.mj_step(model, data, nstep=self.cfg.decimation)

    def step(self, actions):
        cfg = self.cfg
        a = actions.detach().cpu().numpy().astype(float).clip(-1, 1)
        t = self.episode_length_buf.numpy() * cfg.step_dt
        moving = t >= cfg.initial_seconds
        low, high = np.array(plug.ACTION_LOW), np.array(plug.ACTION_HIGH)
        if cfg.translation_control == "delta":
            requested = np.clip(self.tcp_target + cfg.translation_step * a[:, :3], low, high)
        else:
            requested = low + (a[:, :3] + 1) * 0.5 * (high - low)
        step = np.clip(requested - self.tcp_target, -cfg.translation_step, cfg.translation_step)
        self.tcp_target += step * moving[:, None]
        q = self.ik.manip_step(
            torch.as_tensor(self.tcp_target), torch.as_tensor(self.manip_target[:, :5])
        ).numpy()
        self.manip_target[:, :5] = np.where(moving[:, None], q, self.manip_target[:, :5])
        self.manip_target = np.clip(self.manip_target, self.limits[:, 0], self.limits[:, 1])
        for i, (_, data) in enumerate(self.worlds):
            data.ctrl[self.ctrl["manipulator"]] = self.manip_target[i]
        list(self.pool.map(self._physics, self.chunks))
        self._sync()
        self.episode_length_buf += 1
        obj = self.object_pose()[:, :3]
        goal = self.goal()
        o, g = torch.as_tensor(obj), torch.as_tensor(goal)
        if cfg.plug_success_rule == "seated_v2":
            instant = plug_v2.seated(o, g).numpy()
        else:
            instant = np.linalg.norm(obj - goal, axis=-1) < plug.SUCCESS_DISTANCE
        self.hold = np.where(instant, self.hold + 1, 0)
        success = self.hold >= 3
        jam = np.zeros(self.num_envs, dtype=bool)
        if cfg.plug_jam_failure:
            jam = plug_v2.jammed(
                o, g, torch.as_tensor(self.offsets), torch.as_tensor(self.socket)
            ).numpy()
        lost = (obj[:, 2] < -0.035) | (np.abs(obj[:, :2]).max(-1) > 0.42)
        failure = jam | lost | ~np.isfinite(self.qpos).all(-1)
        timeout = self.episode_length_buf.numpy() >= self.max_episode_length
        terminated = success | failure
        reward = self._reward(o, g, success, jam)
        dones = terminated | timeout
        xy_err = np.linalg.norm(obj[:, :2] - goal[:, :2], axis=-1)
        height = obj[:, 2] - goal[:, 2]
        extras = {
            "time_outs": torch.as_tensor(timeout & ~terminated),
            "episode": {
                "done": dones.copy(),
                "success": success.copy(),
                "jam": jam.copy(),
                "timeout": (timeout & ~terminated).copy(),
                "xy_error": xy_err,
                "height": height,
                "error_3d": np.linalg.norm(obj - goal, axis=-1),
                "variant": self.variant_index.copy(),
            },
        }
        self.reset(np.flatnonzero(dones))
        self.obs = self._observe()
        return (
            self.obs,
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(dones),
            extras,
        )

    def _reward(self, obj, goal, success, jam):
        profile = self.cfg.reward_profile
        if profile == "plug_insertion":
            from .plug_learning import insertion_reward

            return insertion_reward(obj, goal).numpy()
        if profile == "plug_seat_v2":
            phi = plug_v2.potential(obj, goal).numpy()
            out = plug_v2.reward(
                torch.as_tensor(phi),
                torch.as_tensor(self.phi),
                torch.as_tensor(self.phi_valid),
                torch.as_tensor(success),
                torch.as_tensor(jam),
            ).numpy()
            self.phi, self.phi_valid = phi, np.ones_like(self.phi_valid)
            return out
        if profile == "progress":
            xy = (obj[:, :2] - goal[:, :2]).norm(dim=-1)
            z = (obj[:, 2] - goal[:, 2]).abs()
            phi = (torch.exp(-xy / 0.035) + torch.exp(-xy / 0.008) * torch.exp(-z / 0.025)).numpy()
            progress = np.where(self.phi_valid, phi - self.phi, 0.0)
            self.phi, self.phi_valid = phi, np.ones_like(self.phi_valid)
            return progress * 3 + success * 10 - 0.005
        raise ValueError(f"Native batch does not implement reward {profile}")
