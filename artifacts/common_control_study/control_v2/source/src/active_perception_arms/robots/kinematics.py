"""Batched torch kinematics for the SO-101 arm.

Provides forward kinematics, geometric Jacobians, and a batched damped-least-
squares IK solver for the 5-joint arm chain, with all chain constants read
from a compiled ``mujoco.MjModel`` (robust to XML edits). Used for reset-time
IK inside event terms (no sim stepping in the loop) and as the CPU/GPU test
surface validated against ``mj_kinematics`` and mink.

Frames: all poses are expressed in the arm *base-body* frame. In the composed
scene each arm's base has a constant world pose; use :class:`BaseFrame` to
convert world-frame targets into base frame before solving.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

# --- small rotation utilities (batched) ---------------------------------------


def quat_wxyz_to_mat(quat: torch.Tensor) -> torch.Tensor:
    """[..., 4] (w, x, y, z) -> [..., 3, 3]."""
    w, x, y, z = quat.unbind(-1)
    two = 2.0
    mat = torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
            two * (x * y + w * z),
            1 - two * (x * x + z * z),
            two * (y * z - w * x),
            two * (x * z - w * y),
            two * (y * z + w * x),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    )
    return mat.reshape(*quat.shape[:-1], 3, 3)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues. ``axis``: [3] unit, ``angle``: [B] -> [B, 3, 3]."""
    b = angle.shape[0]
    k = skew(axis.expand(b, 3))
    s = torch.sin(angle).view(b, 1, 1)
    c = torch.cos(angle).view(b, 1, 1)
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device).expand(b, 3, 3)
    return eye + s * k + (1.0 - c) * (k @ k)


def skew(v: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> [..., 3, 3] cross-product matrix."""
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        (
            zero,
            -v[..., 2],
            v[..., 1],
            v[..., 2],
            zero,
            -v[..., 0],
            -v[..., 1],
            v[..., 0],
            zero,
        ),
        dim=-1,
    ).reshape(*v.shape[:-1], 3, 3)


def mat_to_rotvec(mat: torch.Tensor) -> torch.Tensor:
    """[B, 3, 3] -> [B, 3] rotation vector (angle * axis)."""
    trace = mat.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos)
    vec = torch.stack(
        (
            mat[..., 2, 1] - mat[..., 1, 2],
            mat[..., 0, 2] - mat[..., 2, 0],
            mat[..., 1, 0] - mat[..., 0, 1],
        ),
        dim=-1,
    )
    # angle/(2 sin(angle)) with the small-angle limit 1/2.
    sin = torch.sin(angle)
    scale = torch.where(sin.abs() > 1e-7, angle / (2.0 * sin), torch.full_like(angle, 0.5))
    return vec * scale.unsqueeze(-1)


def pan_tilt_to_axis(pan: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
    """World-frame optical axis for pan/tilt; positive tilt looks down."""
    ct = torch.cos(tilt)
    return torch.stack((ct * torch.cos(pan), ct * torch.sin(pan), -torch.sin(tilt)), dim=-1)


def axis_to_pan_tilt(axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pan_tilt_to_axis` (axis need not be normalized)."""
    a = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    pan = torch.atan2(a[..., 1], a[..., 0])
    tilt = torch.atan2(-a[..., 2], a[..., :2].norm(dim=-1))
    return pan, tilt


# --- chain extraction ----------------------------------------------------------


@dataclass(frozen=True)
class _Link:
    """One jointed link: constant transform from the previous jointed link's
    frame to this body's frame, plus the hinge joint (anchor, axis) in this
    body's frame."""

    rel_pos: torch.Tensor  # [3]
    rel_mat: torch.Tensor  # [3, 3]
    jnt_pos: torch.Tensor  # [3]
    jnt_axis: torch.Tensor  # [3]


class SO101Chain:
    """FK/Jacobian/IK for one site on the SO-101 arm chain.

    Built from a compiled model of the *standalone* arm (or an attached copy —
    pass the name prefix). Poses are in the arm base-body frame.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str,
        joint_names: tuple[str, ...],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.joint_names = joint_names

        def t(x) -> torch.Tensor:
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"site {site_name!r} not in model")
        joint_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name!r} not in model")
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"joint {name!r} is not a hinge")
            joint_ids.append(jid)
        jointed_bodies = [int(model.jnt_bodyid[j]) for j in joint_ids]

        # Walk from the base (parent of the first jointed body) down to the site,
        # accumulating fixed transforms between jointed bodies.
        self.base_body_id = int(model.body_parentid[jointed_bodies[0]])

        # Ancestor chain of the site's body, base -> site body.
        path = []
        b = int(model.site_bodyid[site_id])
        while b != self.base_body_id:
            path.append(b)
            b = int(model.body_parentid[b])
            if b == 0 and self.base_body_id != 0:
                raise ValueError("site is not a descendant of the chain base")
        path.reverse()
        for jb in jointed_bodies:
            if jb not in path:
                raise ValueError("chain joints must lie on the ancestor path of the site")

        self.links: list[_Link] = []
        acc_pos = np.zeros(3)
        acc_mat = np.eye(3)
        for body in path:
            bp = model.body_pos[body].copy()
            bm = np.zeros(9)
            mujoco.mju_quat2Mat(bm, model.body_quat[body])
            bm = bm.reshape(3, 3)
            acc_pos = acc_pos + acc_mat @ bp
            acc_mat = acc_mat @ bm
            if body in jointed_bodies:
                j = joint_ids[jointed_bodies.index(body)]
                self.links.append(
                    _Link(
                        rel_pos=t(acc_pos),
                        rel_mat=t(acc_mat),
                        jnt_pos=t(model.jnt_pos[j].copy()),
                        jnt_axis=t(model.jnt_axis[j].copy()),
                    )
                )
                acc_pos = np.zeros(3)
                acc_mat = np.eye(3)
        # Fixed tail: last jointed body frame -> site frame.
        sp = model.site_pos[site_id].copy()
        sm = np.zeros(9)
        mujoco.mju_quat2Mat(sm, model.site_quat[site_id])
        self.site_pos = t(acc_pos + acc_mat @ sp)
        self.site_mat = t(acc_mat @ sm.reshape(3, 3))

        lo = model.jnt_range[joint_ids, 0]
        hi = model.jnt_range[joint_ids, 1]
        self.joint_low = t(lo)
        self.joint_high = t(hi)

    @property
    def n(self) -> int:
        return len(self.links)

    def fk(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, n] -> site pos [B, 3], site rot [B, 3, 3] (base frame)."""
        pos, mat, _, _ = self._fk_full(q)
        return pos, mat

    def _fk_full(self, q: torch.Tensor):
        b = q.shape[0]
        p = torch.zeros(b, 3, dtype=self.dtype, device=self.device)
        r = torch.eye(3, dtype=self.dtype, device=self.device).expand(b, 3, 3).contiguous()
        axes = []
        anchors = []
        for i, link in enumerate(self.links):
            p = p + (r @ link.rel_pos)
            r = r @ link.rel_mat
            anchor = p + (r @ link.jnt_pos)
            axis = r @ link.jnt_axis
            rj = axis_angle_to_mat(link.jnt_axis, q[:, i])
            # Rotation about the joint anchor within the body frame.
            p = anchor + (r @ rj) @ (-link.jnt_pos)
            r = r @ rj
            axes.append(axis)
            anchors.append(anchor)
        site_p = p + r @ self.site_pos
        site_r = r @ self.site_mat
        return site_p, site_r, torch.stack(axes, dim=1), torch.stack(anchors, dim=1)

    def fk_jac(self, q: torch.Tensor):
        """FK plus geometric Jacobians: returns (pos, rot, jacp [B,3,n], jacr [B,3,n])."""
        pos, rot, axes, anchors = self._fk_full(q)
        lever = pos.unsqueeze(1) - anchors  # [B, n, 3]
        jacp = torch.cross(axes, lever, dim=-1).transpose(1, 2)
        jacr = axes.transpose(1, 2)
        return pos, rot, jacp, jacr

    def clamp(self, q: torch.Tensor) -> torch.Tensor:
        return q.clamp(self.joint_low, self.joint_high)

    def solve_ik(
        self,
        target_pos: torch.Tensor,
        q0: torch.Tensor,
        *,
        target_rot: torch.Tensor | None = None,
        target_axis: torch.Tensor | None = None,
        site_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
        pos_weight: float = 1.0,
        rot_weight: float = 1.0,
        damping: float = 0.05,
        iters: int = 60,
        max_dq: float = 0.5,
    ) -> torch.Tensor:
        """Batched DLS IK in the base frame.

        ``target_rot`` ([B, 3, 3]) adds a full orientation task; ``target_axis``
        ([B, 3], unit) instead aligns the site-frame direction ``site_axis``
        (default -z: the camera optical-axis convention; the jaw direction is site
        +x) with the given world direction — a rank-2 task leaving the remaining
        rotation to the kinematics. At most one of the two may be set.
        """
        if target_rot is not None and target_axis is not None:
            raise ValueError("pass either target_rot or target_axis, not both")
        q = q0.clone()
        lam2 = damping * damping
        eye = torch.eye(self.n, dtype=self.dtype, device=self.device)
        u = torch.tensor(site_axis, dtype=self.dtype, device=self.device)
        for _ in range(iters):
            pos, rot, jacp, jacr = self.fk_jac(q)
            res = [pos_weight * (target_pos - pos)]
            rows = [pos_weight * jacp]
            if target_rot is not None:
                res.append(rot_weight * mat_to_rotvec(target_rot @ rot.transpose(1, 2)))
                rows.append(rot_weight * jacr)
            elif target_axis is not None:
                a_cur = (rot @ u.unsqueeze(-1)).squeeze(-1)
                res.append(rot_weight * (target_axis - a_cur))
                rows.append(rot_weight * (-skew(a_cur) @ jacr))
            e = torch.cat(res, dim=-1).unsqueeze(-1)  # [B, m, 1]
            jac = torch.cat(rows, dim=-2)  # [B, m, n]
            jt = jac.transpose(1, 2)
            dq = torch.linalg.solve(jt @ jac + lam2 * eye, jt @ e).squeeze(-1)
            q = self.clamp(q + dq.clamp(-max_dq, max_dq))
        return q

    def solve_ik_staged(
        self,
        target_pos: torch.Tensor,
        target_rot: torch.Tensor,
        *,
        site_axis: tuple[float, float, float],
        posture: torch.Tensor | None = None,
        iters_axis: int = 80,
        iters_full: int = 60,
        damping: float = 0.05,
    ) -> torch.Tensor:
        """Full-pose IK via a two-stage solve that avoids DLS local minima.

        Stage A solves position + alignment of the site-frame ``site_axis`` with
        its target direction (``target_rot @ site_axis``) — a well-conditioned,
        exactly-determined task. Stage B refines to the full orientation from the
        stage-A configuration. The pan joint is seeded with the target azimuth.
        """
        b = target_pos.shape[0]
        if posture is None:
            posture = torch.tensor([0.0, -0.6, 1.2, 0.9, 0.0], dtype=self.dtype, device=self.device)
        q0 = posture.expand(b, self.n).clone()
        q0[:, 0] = torch.atan2(target_pos[:, 1], target_pos[:, 0]).clamp(
            self.joint_low[0], self.joint_high[0]
        )
        u = torch.tensor(site_axis, dtype=self.dtype, device=self.device)
        a_des = (target_rot @ u.unsqueeze(-1)).squeeze(-1)
        q_axis = self.solve_ik(
            target_pos,
            q0,
            target_axis=a_des,
            site_axis=site_axis,
            iters=iters_axis,
            damping=damping,
        )
        return self.solve_ik(
            target_pos, q_axis, target_rot=target_rot, iters=iters_full, damping=damping
        )


# --- world <-> base frame ------------------------------------------------------


class BaseFrame:
    """Constant world pose of an arm base; converts targets world <-> base."""

    def __init__(
        self,
        pos: tuple[float, float, float],
        yaw: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.pos = torch.tensor(pos, dtype=dtype, device=device)
        c, s = np.cos(yaw), np.sin(yaw)
        self.mat = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device
        )

    def pos_to_base(self, p_world: torch.Tensor) -> torch.Tensor:
        return (p_world - self.pos) @ self.mat  # R^T p == p @ R

    def pos_to_world(self, p_base: torch.Tensor) -> torch.Tensor:
        return p_base @ self.mat.transpose(0, 1) + self.pos

    def vec_to_base(self, v_world: torch.Tensor) -> torch.Tensor:
        return v_world @ self.mat

    def vec_to_world(self, v_base: torch.Tensor) -> torch.Tensor:
        return v_base @ self.mat.transpose(0, 1)

    def mat_to_base(self, r_world: torch.Tensor) -> torch.Tensor:
        return self.mat.transpose(0, 1) @ r_world

    def mat_to_world(self, r_base: torch.Tensor) -> torch.Tensor:
        return self.mat @ r_base


def grasp_site_rotation(
    model: mujoco.MjModel,
    tcp_site: str = "gripperframe",
    yaw: float = 0.0,
    prefix: str = "",
) -> np.ndarray:
    """World rotation target for the TCP site with the jaw pointing straight down.

    The jaw extends along the gripper body's -z axis, so "peg down" pins the
    body's +z to world +z; ``yaw`` then rotates the jaw plane about the world
    vertical. Returns R_site = Rz(yaw) @ R_site_in_gripper_body (3x3).
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + tcp_site)
    if site_id < 0:
        raise ValueError(f"site {prefix + tcp_site!r} not in model")
    sm = np.zeros(9)
    mujoco.mju_quat2Mat(sm, model.site_quat[site_id])
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ sm.reshape(3, 3)
