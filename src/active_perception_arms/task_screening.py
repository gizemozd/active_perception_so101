"""Pre-training viewpoint screen; visibility prototypes, never task-success claims.

Uses the supplied SO-101 geometry, unrestricted fixed cameras, actual camera-arm
FK, MuJoCo visual-geometry ray tests, validation-only view selection, and explicit
information epochs. Prototype pose changes are prescribed interventions; they
are not yet contact-driven manipulation environments.
"""

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from .config import FOVY, JOINTS, Experiment, static_candidates
from .native import Kinematics, calibration
from .scenes import box, look_at_quat, native_spec, yaw_quat


@dataclass(frozen=True)
class Candidate:
    name: str
    mechanism: str
    housing: bool = False
    changing_aperture: bool = False
    fresh: bool = False
    phases: str = "same"
    cap: bool = True


CANDIDATES = (
    Candidate("open_reference", "Uncovered target; conventional sensing control"),
    Candidate("static_recess", "A recessed insert with one persistent viewing aperture", True),
    Candidate(
        "recess_with_slip",
        "Fresh insert offset after contact, unchanged aperture",
        True,
        fresh=True,
    ),
    Candidate(
        "shuttered_insertion",
        "Fresh insert offset and independently changing side aperture",
        True,
        True,
        True,
    ),
    Candidate(
        "two_site_seating",
        "Fresh checks at two distinct sites with differently oriented apertures",
        True,
        False,
        True,
        "opposed",
    ),
    Candidate(
        "open_slot_push",
        "Object displacement under an open longitudinal tool slot",
        True,
        False,
        True,
        "translate",
        False,
    ),
)
CENTER = np.array([-0.025, 0.0, 0.015])
GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)


def rotation_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class ScreenScene:
    def __init__(self, candidate):
        self.candidate = candidate
        self.base_task = "push" if candidate.phases == "translate" else "plug"
        cfg = Experiment(
            task=self.base_task, condition="active", randomize=False, occlusion="clean", num_envs=1
        )
        spec = native_spec(cfg, marker_prototype=True)
        if self.base_task == "push":
            spec.delete(spec.geom("table/partition"))
        # Retain the actual gripper and its pregrasped plug as natural top occlusion.
        # The original socket is hidden; this separate insert is the query target.
        feature = spec.worldbody.add_body(name="screen_feature", mocap=True)
        box(
            feature,
            "screen_insert",
            (0, 0, 0),
            (0.006, 0.006, 0.002),
            (0.04, 0.85, 0.3, 1),
            collision=False,
        )
        if candidate.housing:
            body = spec.worldbody.add_body(name="screen_housing", mocap=True)
            # 120 x 110 mm housing with a narrow front observation window.
            box(
                body,
                "housing_back",
                (0, 0.055, 0.030),
                (0.060, 0.004, 0.030),
                (0.26, 0.35, 0.45, 1),
            )
            for side in (-1, 1):
                box(
                    body,
                    f"housing_side_{side}",
                    (side * 0.060, 0, 0.030),
                    (0.004, 0.055, 0.030),
                    (0.26, 0.35, 0.45, 1),
                )
                box(
                    body,
                    f"front_jamb_{side}",
                    (side * 0.036, -0.055, 0.030),
                    (0.024, 0.004, 0.030),
                    (0.32, 0.41, 0.52, 1),
                )
            box(
                body, "front_sill", (0, -0.055, 0.004), (0.012, 0.004, 0.004), (0.32, 0.41, 0.52, 1)
            )
            box(
                body,
                "front_lintel",
                (0, -0.055, 0.056),
                (0.012, 0.004, 0.004),
                (0.32, 0.41, 0.52, 1),
            )
            # Broad top access for the gripper/plug; open-slot variant leaves a
            # longitudinal view available as a strong counterexample to movement.
            hole = 0.030 if candidate.cap else 0.052
            for side in (-1, 1):
                box(
                    body,
                    f"top_side_{side}",
                    (side * (0.060 + hole) / 2, 0, 0.060),
                    ((0.060 - hole) / 2, 0.055, 0.003),
                    (0.32, 0.41, 0.52, 1),
                )
                if candidate.cap:
                    box(
                        body,
                        f"top_end_{side}",
                        (0, side * 0.043, 0.060),
                        (hole, 0.012, 0.003),
                        (0.32, 0.41, 0.52, 1),
                    )
            if candidate.phases == "opposed":
                other = spec.worldbody.add_body(name="second_housing", mocap=True)
                for geom in list(body.geoms):
                    box(other, "second_" + geom.name, geom.pos, geom.size, geom.rgba)
            if not candidate.cap:
                for geom in list(body.geoms):
                    if geom.name.startswith("front_"):
                        spec.delete(geom)
        spec.worldbody.add_camera(
            name="screen_fixed",
            pos=(-0.15, -0.22, 0.22),
            quat=look_at_quat((-0.15, -0.22, 0.22), CENTER),
            fovy=FOVY,
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.ik = Kinematics(self.model)
        self.home = calibration(self.base_task, marker_prototype=True)
        self.qadr = {
            name: np.array([self.model.joint(name + "/" + j).qposadr[0] for j in JOINTS])
            for name in ("manipulator", "camera_arm")
        }
        self.ctrlids = {
            name: np.array([self.model.actuator(name + "/" + j).id for j in JOINTS])
            for name in self.qadr
        }
        for name in self.qadr:
            self.data.qpos[self.qadr[name]] = self.home[name]
            self.data.ctrl[self.ctrlids[name]] = self.home[name]
        self.object_adr = self.model.joint("object/free").qposadr[0]
        self.data.qpos[self.object_adr : self.object_adr + 7] = (
            self.home["plug_pose"] if self.base_task == "plug" else (0, 0, 0.009, 1, 0, 0, 0)
        )
        for body in ("fixture/fixture", "occluder/panel"):
            self.data.mocap_pos[self.model.body(body).mocapid[0]] = (0, 0, -1)
        self.feature_id = self.model.body("screen_feature").mocapid[0]
        self.feature_geom = self.model.geom("screen_insert").id
        self.housing_id = (
            self.model.body("screen_housing").mocapid[0] if candidate.housing else None
        )
        self.second_housing_id = (
            self.model.body("second_housing").mocapid[0] if candidate.phases == "opposed" else None
        )
        self.arm_bodies = {
            name: {
                i
                for i in range(self.model.nbody)
                if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(
                    name + "/"
                )
            }
            for name in self.qadr
        }
        self.sample_offsets = np.array(
            [
                (x, y, z)
                for z in (-0.002, 0.002)
                for x in np.linspace(-0.0055, 0.0055, 5)
                for y in np.linspace(-0.0055, 0.0055, 5)
            ]
        )
        self.set_episode(0, 0)

    def set_episode(self, seed, phase):
        rng = np.random.default_rng(seed)
        heading = rng.choice([-1.0, 1.0], size=3)
        offsets = rng.uniform(-0.005, 0.005, size=(3, 2))
        jitter = rng.uniform(-0.004, 0.004, size=2)
        candidate = self.candidate
        direction = heading[phase if candidate.changing_aperture else 0]
        yaw = -0.60 + direction * 0.38
        center = CENTER.copy()
        center[:2] += jitter
        if candidate.phases == "opposed":
            centers = [CENTER + [-0.040, -0.065, 0], CENTER + [0.050, 0.065, 0]]
            yaws = [-0.98, -0.22]
            for idx, mid in enumerate((self.housing_id, self.second_housing_id)):
                self.data.mocap_pos[mid] = centers[idx] + np.r_[jitter, -0.010]
                self.data.mocap_quat[mid] = yaw_quat(yaws[idx])
            center = centers[int(phase == 2)] + np.r_[jitter, 0.0]
            yaw = yaws[int(phase == 2)]
        if candidate.phases == "translate":
            center[1] += (phase - 1) * 0.025
            center[2] = 0.020
            yaw = -0.60
        R = rotation_z(yaw)
        epoch = phase if candidate.fresh else 0
        local = np.r_[offsets[epoch], 0.0]
        self.data.mocap_pos[self.feature_id] = center + R @ local
        self.data.mocap_quat[self.feature_id] = yaw_quat(yaw)
        if self.housing_id is not None and self.second_housing_id is None:
            self.data.mocap_pos[self.housing_id] = center - [0, 0, 0.010]
            self.data.mocap_quat[self.housing_id] = yaw_quat(yaw)
        # Move the held plug into the top opening after the initial observation.
        tcp = center + [0, -0.0015, 0.055 if phase else 0.095]
        if candidate.phases == "translate":
            # Pushing control: a tool at the rear, with the object ahead in a slot.
            tcp = center + R @ np.array([0, -0.035, 0.030])
            self.data.qpos[self.object_adr : self.object_adr + 3] = np.r_[
                self.data.mocap_pos[self.feature_id][:2], 0.009
            ]
            self.data.qpos[self.object_adr + 3 : self.object_adr + 7] = yaw_quat(yaw)
        self.set_manip_tcp(tcp)
        self.epoch = epoch
        self.yaw = yaw
        self.feature_center = self.data.mocap_pos[self.feature_id].copy()
        self.points = self.feature_center + self.sample_offsets @ R.T
        self.expected_tcp = tcp

    def set_manip_tcp(self, tcp):
        q = self.ik.manip(tcp, self.home["manipulator"][:5])
        self.data.qpos[self.qadr["manipulator"][:5]] = q
        self.data.ctrl[self.ctrlids["manipulator"][:5]] = q
        mujoco.mj_forward(self.model, self.data)
        # Enforce the exact pregrasp transform in frozen snapshots.
        from .scenes import grasp_relpose

        if self.base_task == "plug":
            p, quat = grasp_relpose()
            body = self.model.body("manipulator/gripper").id
            self.data.qpos[self.object_adr : self.object_adr + 3] = (
                self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ p
            )
            mujoco.mju_mulQuat(
                self.data.qpos[self.object_adr + 3 : self.object_adr + 7],
                self.data.xquat[body],
                quat,
            )
        mujoco.mj_forward(self.model, self.data)

    def refresh_points(self):
        """Update ray targets after a counterfactual feature-only intervention."""
        mujoco.mj_forward(self.model, self.data)
        self.feature_center = self.data.mocap_pos[self.feature_id].copy()
        self.points = self.feature_center + self.sample_offsets @ rotation_z(self.yaw).T

    def set_camera_arm(self, q):
        self.data.qpos[self.qadr["camera_arm"][:5]] = q
        self.data.ctrl[self.ctrlids["camera_arm"][:5]] = q
        mujoco.mj_forward(self.model, self.data)

    def camera_view(self, name):
        cid = self.model.camera(name).id
        return self.data.cam_xpos[cid].copy(), self.data.cam_xmat[cid].reshape(3, 3).copy()

    def visibility(self, position, matrix, width=96, height=72):
        local = (self.points - position) @ matrix
        depth = -local[:, 2]
        f = height / (2 * np.tan(np.radians(FOVY) / 2))
        uv = np.column_stack(
            [
                f * local[:, 0] / np.maximum(depth, 1e-8) + width / 2,
                height / 2 - f * local[:, 1] / np.maximum(depth, 1e-8),
            ]
        )
        in_frame = (
            (depth > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        rays = self.points - position
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        ids = np.full(len(rays), -1, dtype=np.int32)
        distances = np.empty(len(rays))
        mujoco.mj_multiRay(
            self.model,
            self.data,
            np.asarray(position, dtype=float),
            rays.ravel(),
            GROUPS,
            True,
            -1,
            ids,
            distances,
            None,
            len(rays),
            3.0,
        )
        visible = in_frame & (ids == self.feature_geom)
        # Probe bins can overestimate rasterized area. Cast rays through actual
        # pixel centers in the projected bounding box before applying the gate.
        pixels = np.unique(np.floor(uv[visible]).astype(int), axis=0).shape[0]
        pixel_count = 0
        if np.all(depth > 0):
            low = np.maximum(np.floor(uv.min(axis=0)).astype(int) - 1, 0)
            high = np.minimum(np.ceil(uv.max(axis=0)).astype(int) + 1, [width - 1, height - 1])
            u, v = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1))
            if u.size:
                local_rays = np.column_stack(
                    [
                        (u.ravel() + 0.5 - width / 2) / f,
                        (height / 2 - v.ravel() - 0.5) / f,
                        -np.ones(u.size),
                    ]
                )
                pixel_rays = local_rays @ matrix.T
                pixel_rays /= np.linalg.norm(pixel_rays, axis=1, keepdims=True)
                pixel_ids = np.full(u.size, -1, dtype=np.int32)
                pixel_distances = np.empty(u.size)
                mujoco.mj_multiRay(
                    self.model,
                    self.data,
                    np.asarray(position, dtype=float),
                    pixel_rays.ravel(),
                    GROUPS,
                    True,
                    -1,
                    pixel_ids,
                    pixel_distances,
                    None,
                    u.size,
                    3.0,
                )
                pixel_count = int((pixel_ids == self.feature_geom).sum())
        return {
            "fraction": float(visible.mean()),
            "sampled_pixels": int(pixels),
            "marker_pixels": pixel_count,
            "visible": bool(pixel_count >= 3),
        }

    def camera_contacts(self):
        return self.external_contacts("camera_arm")

    def external_contacts(self, arm):
        ids = self.arm_bodies[arm]
        severe = []
        for contact in self.data.contact:
            a, b = self.model.geom_bodyid[contact.geom]
            if contact.dist < -0.001 and ((a in ids) != (b in ids)):
                severe.append(
                    {
                        "depth_m": float(-contact.dist),
                        "geoms": [self.model.geom(int(g)).name for g in contact.geom],
                    }
                )
        return severe

    def render(self, path, view=None, camera="overview", size=(960, 540)):
        if view is not None:
            cid = self.model.camera("screen_fixed").id
            position, matrix = view
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, matrix.ravel())
            self.model.cam_pos[cid] = position
            self.model.cam_quat[cid] = quat
            mujoco.mj_forward(self.model, self.data)
            camera = "screen_fixed"
        renderer = mujoco.Renderer(self.model, width=size[0], height=size[1])
        option = mujoco.MjvOption()
        option.geomgroup[:] = GROUPS
        option.sitegroup[:] = 0
        try:
            renderer.update_scene(self.data, camera=camera, scene_option=option)
            rgb = renderer.render().copy()
            if path:
                Image.fromarray(rgb).save(path)
            return rgb
        finally:
            renderer.close()


def camera_catalog(scene):
    entries = []
    for position in itertools.product(
        (-0.26, -0.18, -0.10, -0.02), (-0.30, -0.22, -0.14, -0.06), (0.10, 0.16, 0.24, 0.32)
    ):
        q = scene.ik.camera(np.array(position), CENTER, scene.home["camera_arm"][:5])
        scene.set_camera_arm(q)
        actual, matrix = scene.camera_view("camera_arm/wrist_cam")
        desired = CENTER - actual
        desired /= np.linalg.norm(desired)
        angle = np.degrees(np.arccos(np.clip(-matrix[:, 2] @ desired, -1, 1)))
        error = np.linalg.norm(actual - position)
        if error < 0.012 and angle < 8 and not scene.camera_contacts():
            entries.append(
                {
                    "position": actual.tolist(),
                    "matrix": matrix.tolist(),
                    "q": q.tolist(),
                    "ik_error_m": float(error),
                    "aim_error_deg": float(angle),
                }
            )
    scene.set_camera_arm(scene.home["camera_arm"][:5])
    return entries


def fixed_catalog(reachable):
    positions = list(static_candidates())
    for radius, height, azimuth in itertools.product(
        (0.20, 0.30, 0.45), (0.12, 0.22, 0.35, 0.50, 0.65), range(-180, 180, 15)
    ):
        angle = np.radians(azimuth)
        positions.append(
            tuple(
                CENTER
                + np.array([radius * np.cos(angle), radius * np.sin(angle), height - CENTER[2]])
            )
        )
    entries = []
    for position in positions:
        quat = look_at_quat(position, CENTER)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quat)
        entries.append({"position": list(position), "matrix": matrix.reshape(3, 3).tolist()})
    # A fixed camera may occupy every reachable active endpoint as well.
    entries.extend({k: e[k] for k in ("position", "matrix")} for e in reachable)
    return entries


def refine_fixed_catalog(fixed, validation, candidate):
    """Refine four spatially separated validation leaders before opening test seeds."""
    scores = knowledge(validation["fixed"], validation["wrist"], candidate.fresh)
    single = knowledge(validation["fixed"], np.zeros_like(validation["wrist"]), candidate.fresh)
    complete = scores.all(axis=1).mean(axis=0)
    coverage = scores.mean(axis=(0, 1))
    alone = single.mean(axis=(0, 1))
    order = sorted(
        range(len(fixed)), key=lambda i: (complete[i], coverage[i], alone[i]), reverse=True
    )
    centers = []
    for idx in order:
        point = np.array(fixed[idx]["position"])
        if all(np.linalg.norm(point - p) > 0.06 for p in centers):
            centers.append(point)
        if len(centers) == 4:
            break
    added = []
    for center in centers:
        for delta in itertools.product((-0.025, 0, 0.025), (-0.025, 0, 0.025), (-0.030, 0, 0.030)):
            if delta == (0, 0, 0):
                continue
            point = center + delta
            quat = look_at_quat(point, CENTER)
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, quat)
            added.append({"position": point.tolist(), "matrix": matrix.reshape(3, 3).tolist()})
    return added


def collect(scene, seeds, fixed, active, width=96, height=72):
    nv, na = len(fixed), len(active)
    fixed_seen = np.zeros((len(seeds), 3, nv), dtype=bool)
    active_seen = np.zeros((len(seeds), 3, na), dtype=bool)
    wrist = np.zeros((len(seeds), 3), dtype=bool)
    active_wrist = np.zeros_like(active_seen)
    feasible = np.zeros_like(active_seen)
    intrusions = []
    tcp_errors = []

    def visible(view):
        return scene.visibility(*view, width=width, height=height)["visible"]

    for row, seed in enumerate(seeds):
        for phase in range(3):
            scene.set_episode(seed, phase)
            scene.set_camera_arm(scene.home["camera_arm"][:5])
            wrist[row, phase] = visible(scene.camera_view("manipulator/wrist_cam"))
            tcp_errors.append(
                float(
                    np.linalg.norm(
                        scene.data.site("manipulator/gripperframe").xpos - scene.expected_tcp
                    )
                )
            )
            collisions = []
            for contact in scene.data.contact:
                names = [scene.model.geom(int(g)).name for g in contact.geom]
                if contact.dist < -0.001 and any(
                    "housing_" in n or "top_" in n or "front_" in n for n in names
                ):
                    collisions.append({"depth_m": float(-contact.dist), "geoms": names})
            if collisions:
                intrusions.append({"seed": seed, "phase": phase, "contacts": collisions})
            for idx, view in enumerate(fixed):
                fixed_seen[row, phase, idx] = visible(
                    (np.array(view["position"]), np.array(view["matrix"]))
                )
            for idx, view in enumerate(active):
                scene.set_camera_arm(np.array(view["q"]))
                if not scene.camera_contacts():
                    feasible[row, phase, idx] = True
                    active_seen[row, phase, idx] = visible(
                        scene.camera_view("camera_arm/wrist_cam")
                    )
                    active_wrist[row, phase, idx] = visible(
                        scene.camera_view("manipulator/wrist_cam")
                    )
    return {
        "fixed": fixed_seen,
        "active": active_seen,
        "wrist": wrist,
        "active_wrist": active_wrist,
        "feasible": feasible,
        "fixture_intrusions": intrusions,
        "max_tcp_error_m": max(tcp_errors),
    }


def knowledge(seen, wrist, fresh):
    combined = seen | wrist[..., None]
    if not fresh:
        combined = np.maximum.accumulate(combined, axis=1)
    return combined[:, 1:]


def best_static_index(knowledge_array):
    """Validation completion, then query coverage, then catalog order."""
    complete = knowledge_array.all(axis=1).mean(axis=0)
    coverage = knowledge_array.mean(axis=(0, 1))
    return max(range(len(complete)), key=lambda i: (complete[i], coverage[i], -i))


def summarize(candidate, validation, test):
    vf, va, vw = validation["fixed"], validation["active"], validation["wrist"]
    tf, ta, tw = test["fixed"], test["active"], test["wrist"]
    vk = knowledge(vf, vw, candidate.fresh)
    tk = knowledge(tf, tw, candidate.fresh)
    index = best_static_index(vk)
    static_index = best_static_index(knowledge(vf, np.zeros_like(vw), candidate.fresh))
    static_only = knowledge(tf, np.zeros_like(tw), candidate.fresh)
    wrist_only = knowledge(np.zeros_like(tf[:, :, :1]), tw, candidate.fresh)
    ak = knowledge(ta | test["active_wrist"], np.zeros_like(tw), candidate.fresh)
    av = knowledge(va | validation["active_wrist"], np.zeros_like(vw), candidate.fresh)
    pair_score = np.einsum("ni,nj->ij", av[:, 0].astype(float), av[:, 1].astype(float)) / len(av)
    scheduled = np.unravel_index(pair_score.argmax(), pair_score.shape)
    if not candidate.fresh:
        scheduled = (best_static_index(av),) * 2
    # A fixed two-view scan can revisit BOTH views at EVERY query. This is much
    # stronger than a single preselected view per phase; motion is audited later.
    scan_score = (
        (av[:, 0, :, None] | av[:, 0, None, :]) & (av[:, 1, :, None] | av[:, 1, None, :])
    ).mean(axis=0)
    scan = np.unravel_index(scan_score.argmax(), scan_score.shape)
    scan_complete = ak[:, :, list(scan)].any(axis=2).all(axis=1)
    fixed_complete = tk[:, :, index].all(axis=1)
    moving_complete = ak.any(axis=2).all(axis=1)
    return {
        "best_static_index": index,
        "best_static_plus_wrist": float(tk[:, :, index].all(axis=1).mean()),
        "best_static_only_index": static_index,
        "best_static_only": float(static_only[:, :, static_index].all(axis=1).mean()),
        "wrist_with_memory": float(wrist_only.all(axis=1).mean()),
        "wrist_without_memory": float(tw[:, 1:].all(axis=1).mean()),
        "static_plus_wrist_without_memory": float((tf[:, :, index] | tw)[:, 1:].all(axis=1).mean()),
        "oracle_initial_fixed_plus_wrist": float(tk.all(axis=1).any(axis=1).mean()),
        "oracle_initial_reachable_plus_wrist": float(ak.all(axis=1).any(axis=1).mean()),
        "best_validation_scheduled_pair": [int(x) for x in scheduled],
        "scheduled_pair_plus_wrist": float(
            (ak[:, 0, scheduled[0]] & ak[:, 1, scheduled[1]]).mean()
        ),
        "best_validation_scan_pair": [int(x) for x in scan],
        "two_view_scan_plus_wrist": float(scan_complete.mean()),
        "oracle_moving_plus_wrist": float(moving_complete.mean()),
        "wait_static_same_state": float(tk[:, :, index].all(axis=1).mean()),
        "static_query_coverage": float(tk[:, :, index].mean()),
        "active_query_coverage": float(ak.any(axis=2).mean()),
        "episodes": len(tw),
        "paired_oracle_moving_vs_static_counts": {
            "both": int((moving_complete & fixed_complete).sum()),
            "moving_only": int((moving_complete & ~fixed_complete).sum()),
            "static_only": int((~moving_complete & fixed_complete).sum()),
            "neither": int((~moving_complete & ~fixed_complete).sum()),
        },
        "fixture_intrusion_snapshot_count": len(test["fixture_intrusions"]),
        "max_tcp_error_m": test["max_tcp_error_m"],
        "test_complete_by_static_view": tk.all(axis=1).mean(axis=0).tolist(),
        "test_complete_by_initial_reachable_view": ak.all(axis=1).mean(axis=0).tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--test-episodes", type=int)
    parser.add_argument("--validation-seed", type=int, default=10000)
    parser.add_argument("--test-seed", type=int, default=30000)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/task_screening"))
    parser.add_argument("--candidates", nargs="+", choices=[c.name for c in CANDIDATES])
    args = parser.parse_args(argv)
    if args.episodes < 1 or (args.test_episodes is not None and args.test_episodes < 1):
        parser.error("episode counts must be positive")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    reference = ScreenScene(CANDIDATES[0])
    active = camera_catalog(reference)
    if not active:
        raise RuntimeError("No feasible camera endpoints")
    fixed = fixed_catalog(active)
    report = {
        "purpose": "pre-training geometric information screen, not task success",
        "visibility_rule": "At least 3 marker pixels from rays through image pixel centers; no learned pose-estimation claim.",
        "policy_resolution": [args.width, args.height],
        "fixed_view_count": len(fixed),
        "active_endpoint_count": len(active),
        "validation_seeds": list(range(args.validation_seed, args.validation_seed + args.episodes)),
        "test_seeds": list(
            range(args.test_seed, args.test_seed + (args.test_episodes or args.episodes))
        ),
        "assumptions": [
            "Prescribed geometry/pose interventions, not contact-driven tasks.",
            "Visible marker proxy is not a pose estimator or task-success metric.",
            "Oracle initial views can use future states; they are upper bounds.",
            "Oracle moving views have privileged state and no travel cost.",
            "Two-view scans revisit both views at every query without travel cost.",
            "Waiting assumes the current housing remains in place.",
            "Fixed-camera search is finite, matched FOV, and includes overhead.",
            "Insertion/seating prototypes use the held-plug footprint; the open-slot control uses the pushing tip and block.",
        ],
        "active_catalog": active,
        "fixed_catalog": fixed,
        "candidates": {},
    }
    if set(report["validation_seeds"]) & set(report["test_seeds"]):
        parser.error("validation and test seeds must be disjoint")
    for candidate in CANDIDATES:
        if args.candidates and candidate.name not in args.candidates:
            continue
        scene = ScreenScene(candidate)
        val = collect(scene, report["validation_seeds"], fixed, active, args.width, args.height)
        refined = refine_fixed_catalog(fixed, val, candidate)
        extra = collect(scene, report["validation_seeds"], refined, [], args.width, args.height)
        val["fixed"] = np.concatenate([val["fixed"], extra["fixed"]], axis=2)
        searched_fixed = fixed + refined
        test = collect(scene, report["test_seeds"], searched_fixed, active, args.width, args.height)
        stats = summarize(candidate, val, test)
        report["candidates"][candidate.name] = {
            "mechanism": candidate.mechanism,
            "fixed_view_count": len(searched_fixed),
            "fixed_catalog": searched_fixed,
            **stats,
            "fixture_intrusions": test["fixture_intrusions"],
        }
        scene.set_episode(args.test_seed, 1)
        best = searched_fixed[stats["best_static_index"]]
        scene.set_camera_arm(scene.home["camera_arm"][:5])
        if not args.no_render:
            scene.render(args.output / f"{candidate.name}_overview.png")
            scene.render(
                args.output / f"{candidate.name}_static.png",
                (np.array(best["position"]), np.array(best["matrix"])),
                size=(args.width, args.height),
            )
        print(
            candidate.name,
            {k: v for k, v in stats.items() if not isinstance(v, (list, dict))},
            flush=True,
        )
        (args.output / "screening.json").write_text(json.dumps(report, indent=2))
    print("Saved", args.output / "screening.json")


if __name__ == "__main__":
    main()
