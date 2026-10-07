"""One set of physical assets for native MuJoCo checks and MjLab/MJWarp."""

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from . import plug
from .config import CAMERA_BASE, CAMERA_BASE_YAW, FOVY, MANIP_BASE, Experiment

ROBOT_XML = Path(__file__).parent / "assets/so101/so101_arm.xml"
OVERVIEW_SIZE = (1920, 1080)
OVERVIEW_POSITION = (-0.92, -0.55, 0.67)
OVERVIEW_TARGET = (-0.15, -0.21, 0.13)


def yaw_quat(angle):
    return (np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2))


def look_at_quat(position, target):
    """Camera local -Z looks at target; +Y is image up."""
    z = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(z @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).ravel())
    return quat


def box(body, name, pos, size, rgba, *, collision=True, mass=None):
    geom = body.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=rgba, group=0
    )
    geom.contype = geom.conaffinity = 1 if collision else 0
    geom.friction = [0.8, 0.005, 0.0001]
    if mass is not None:
        geom.mass = mass
    return geom


def arm_spec():
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    for key in list(spec.keys):
        key.delete()
    spec.camera("wrist_cam").fovy = FOVY
    # Preserve all mechanical collisions. Visual groups exclude collision proxies.
    for material in spec.materials:
        if material.rgba[0] > 0.5:
            material.rgba = [0.92, 0.92, 0.94, 1.0]
    box(spec.body("base"), "riser", (0, 0, -0.025), (0.05, 0.05, 0.025), (0.2, 0.2, 0.23, 1))
    return spec


def object_spec(task, variant="xm"):
    if task == "plug":
        return plug.object_spec(variant)
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="free")
    if task == "marker_prototype":
        # Pregrasped rigid plug: keyed rectangular pin and grasp collar.
        box(body, "housing", (0, 0, 0.020), (0.023, 0.018, 0.010), (0.9, 0.9, 0.95, 1), mass=0.025)
        box(body, "pin", (0, 0, 0), (0.006, 0.004, 0.010), (0.85, 0.6, 0.15, 1), mass=0.004)
        box(body, "collar", (0, 0, 0.041), (0.013, 0.017, 0.011), (0.9, 0.9, 0.95, 1), mass=0.008)
        body.add_site(name="tip", pos=(0, 0, -0.01), size=[0.001] * 3, group=5)
    else:
        size = (0.012, 0.012, 0.012) if task == "transfer" else (0.015, 0.012, 0.009)
        box(body, "block", (0, 0, 0), size, (0.95, 0.25, 0.04, 1), mass=0.025)
        body.add_site(name="tip", pos=(0, 0, 0), size=[0.001] * 3, group=5)
    return spec


def fixture_spec(task, clearance=0.002):
    if task == "plug":
        return plug.fixture_spec(clearance)
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="fixture", mocap=True)
    if task == "marker_prototype":
        hx, hy = 0.006 + clearance, 0.004 + clearance
        sx, sy, h = 0.035, 0.027, 0.01
        for name, p, s in (
            ("left", (-(sx + hx) / 2, 0, h), ((sx - hx) / 2, sy, h)),
            ("right", ((sx + hx) / 2, 0, h), ((sx - hx) / 2, sy, h)),
            ("front", (0, -(sy + hy) / 2, h), (hx, (sy - hy) / 2, h)),
            ("back", (0, (sy + hy) / 2, h), (hx, (sy - hy) / 2, h)),
        ):
            box(body, name, p, s, (0.20, 0.55, 0.75, 1))
        body.add_site(name="goal", pos=(0, 0, 0.003), size=[0.002] * 3, group=5)
    elif task == "transfer":
        box(body, "floor", (0.025, 0, 0.023), (0.075, 0.055, 0.023), (0.25, 0.40, 0.50, 1))
        box(body, "back", (0.103, 0, 0.190), (0.003, 0.061, 0.190), (0.32, 0.40, 0.48, 1))
        box(body, "side", (0.025, 0.064, 0.190), (0.081, 0.003, 0.190), (0.32, 0.40, 0.48, 1))
        box(body, "roof", (0.025, 0, 0.380), (0.081, 0.067, 0.003), (0.32, 0.40, 0.48, 1))
        box(
            body,
            "marker",
            (0, 0, 0.0462),
            (0.023, 0.023, 0.0002),
            (0.1, 0.8, 0.3, 1),
            collision=False,
        )
        body.add_site(name="goal", pos=(0, 0, 0.058), size=[0.002] * 3, group=5)
    else:
        box(
            body,
            "marker",
            (0, 0, 0.0003),
            (0.026, 0.026, 0.0003),
            (0.1, 0.8, 0.3, 1),
            collision=False,
        )
        body.add_site(name="goal", pos=(0, 0, 0.009), size=[0.002] * 3, group=5)
    return spec


def occluder_spec():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="panel", mocap=True, pos=(0, 0, -1))
    # Optical intervention: isolate missing information from contact disturbances.
    # Real fixture walls and both robots retain their mechanical collisions.
    box(body, "panel", (0, 0, 0), (0.040, 0.004, 0.070), (0.45, 0.24, 0.50, 1), collision=False)
    return spec


def table_spec(task):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table")
    box(body, "top", (0, 0.15, -0.022), (0.45, 0.50, 0.022), (0.24, 0.29, 0.34, 1))
    if task == "transfer":
        x, y = -0.025, -0.045
        box(body, "tray_floor", (x, y, 0.022), (0.035, 0.033, 0.022), (0.7, 0.64, 0.48, 1))
        box(body, "tray_wall", (x - 0.037, y, 0.034), (0.002, 0.033, 0.034), (0.7, 0.64, 0.48, 1))
    if task == "push":
        box(body, "partition", (0.036, 0.003, 0.025), (0.004, 0.02, 0.025), (0.65, 0.57, 0.35, 1))
    return spec


@lru_cache
def grasp_relpose():
    # Housing origin lies 45 mm below the TCP; gripper body yaw is -pi/2.
    model = arm_spec().compile()
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p_rel = model.site("gripperframe").pos - rotation.T @ np.array([0, -0.0015, 0.045])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.T.ravel())
    return p_rel, quat


def finish_scene(spec, cfg: Experiment, marker_prototype=False):
    spec.option.timestep = cfg.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.iterations = 30
    spec.option.ls_iterations = 10
    spec.visual.global_.offwidth, spec.visual.global_.offheight = OVERVIEW_SIZE
    spec.visual.headlight.ambient = [0.28, 0.28, 0.28]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.15, 0.15, 0.15]
    spec.worldbody.add_light(
        name="task_light",
        pos=(-0.3, -0.4, 1.5),
        dir=(0.15, 0.2, -1),
        diffuse=(0.7, 0.68, 0.64),
        castshadow=True,
    )
    spec.worldbody.add_light(
        name="fill_light",
        pos=(0.5, 0.6, 1.2),
        dir=(-0.3, -0.3, -1),
        diffuse=(0.25, 0.28, 0.32),
        castshadow=False,
    )
    spec.worldbody.add_camera(
        name="fixed",
        pos=cfg.fixed_position,
        quat=look_at_quat(cfg.fixed_position, cfg.fixed_lookat),
        fovy=FOVY,
    )
    # Observer camera for full-resolution videos. Policy sensors are registered
    # separately, so this adds no high-resolution rendering to the training loop.
    plug_scene = cfg.task == "plug" and not marker_prototype
    spec.worldbody.add_camera(
        name="overview",
        pos=(-0.80, -0.75, 0.67) if plug_scene else OVERVIEW_POSITION,
        quat=look_at_quat((-0.80, -0.75, 0.67), (-0.06, 0.20, 0.09))
        if plug_scene
        else look_at_quat(OVERVIEW_POSITION, OVERVIEW_TARGET),
        fovy=42,
    )
    if plug_scene:
        _workbench_visuals(spec)
        spec.worldbody.add_camera(
            name="task_detail",
            pos=(0.18, -0.25, 0.16),
            quat=look_at_quat((0.18, -0.25, 0.16), (0.015, 0.025, 0.04)),
            fovy=34,
        )
    if cfg.task == "push":
        # A small rigid pushing tip keeps contact below the bare jaw edges.
        # It is a physical, visible tool, shared by native and Warp scenes.
        tcp = spec.site("manipulator/gripperframe").pos
        tip = spec.body("manipulator/gripper").add_geom(
            name="pushing_tip",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=tcp + np.array([0, 0, -0.018]),
            size=(0.008, 0.020, 0),
            rgba=(0.1, 0.7, 0.7, 1),
            mass=0.005,
            group=0,
        )
        tip.friction = [0.5, 0.005, 0.0001]
    if cfg.task == "plug":
        p, q = grasp_relpose()
        eq = spec.add_equality(
            name="pregrasp",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1="manipulator/gripper",
            name2="object/object",
        )
        eq.data[0:3] = 0.0
        eq.data[3:6], eq.data[6:10], eq.data[10] = p, q, 1.0
        eq.solref = [0.01 if plug_scene else 0.008, 1.0]
        if marker_prototype:
            for body in ("manipulator/gripper", "manipulator/moving_jaw_so101_v1"):
                spec.add_exclude(bodyname1=body, bodyname2="object/object")


def _workbench_visuals(spec):
    """Static workshop detail; no variant labels or added task collisions."""
    spec.add_texture(
        name="workshop_background",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        width=256,
        height=1536,
        rgb1=(0.42, 0.48, 0.54),
        rgb2=(0.20, 0.23, 0.27),
    )
    spec.add_texture(
        name="bench_texture",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        width=256,
        height=256,
        rgb1=(0.22, 0.27, 0.27),
        rgb2=(0.23, 0.28, 0.28),
        mark=mujoco.mjtMark.mjMARK_RANDOM,
        markrgb=(0.25, 0.30, 0.30),
        random=0.035,
    )
    mat = spec.add_material(
        name="bench_finish", rgba=(1, 1, 1, 1), specular=0.08, shininess=0.1, texrepeat=(4, 4)
    )
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "bench_texture"
    spec.geom("table/top").material = "bench_finish"
    spec.geom("table/top").rgba = (1, 1, 1, 1)
    # Extend support under the restored north camera base, without changing top z.
    spec.geom("table/top").size = (0.52, 0.65, 0.022)
    spec.geom("table/top").pos = (0, 0.20, -0.022)
    room = spec.worldbody.add_body(name="workshop")
    room.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, -0.76),
        size=(3, 3, 0.05),
        rgba=(0.25, 0.28, 0.32, 1),
        contype=0,
        conaffinity=0,
        group=2,
    )
    for x in (-0.46, 0.46):
        for y in (-0.39, 0.79):
            box(
                room,
                f"leg_{x}_{y}",
                (x, y, -0.39),
                (0.025, 0.025, 0.35),
                (0.12, 0.14, 0.16, 1),
                collision=False,
            )
    # Small flush fasteners on the fixture are identical in all variants.
    fixture = spec.body("fixture/fixture")
    for x in (-0.042, 0.042):
        for y in (-0.027, 0.027):
            fixture.add_geom(
                name=f"socket_screw_{x}_{y}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=(x, y, 0.01405),
                size=(0.0023, 0.00005, 0),
                rgba=(0.22, 0.24, 0.26, 1),
                mass=0,
                contype=0,
                conaffinity=0,
                group=2,
            )


def native_spec(cfg: Experiment, *, marker_prototype=False):
    spec = mujoco.MjSpec()
    task = "marker_prototype" if marker_prototype and cfg.task == "plug" else cfg.task
    cam_base = CAMERA_BASE if marker_prototype else cfg.camera_base
    cam_yaw = CAMERA_BASE_YAW if marker_prototype else cfg.camera_base_yaw
    for name, child, pos, quat in (
        ("manipulator", arm_spec(), MANIP_BASE, (1, 0, 0, 0)),
        ("camera_arm", arm_spec(), cam_base, yaw_quat(cam_yaw)),
        ("object", object_spec(task, cfg.plug_variant or "xm"), (0, 0, 0), (1, 0, 0, 0)),
        ("fixture", fixture_spec(task, cfg.clearance), (0, 0, 0), (1, 0, 0, 0)),
        ("occluder", occluder_spec(), (0, 0, 0), (1, 0, 0, 0)),
        ("table", table_spec(cfg.task), (0, 0, 0), (1, 0, 0, 0)),
    ):
        frame = spec.worldbody.add_frame(pos=pos, quat=quat)
        spec.attach(child, prefix=name + "/", frame=frame)
    finish_scene(spec, cfg, marker_prototype=marker_prototype)
    return spec
