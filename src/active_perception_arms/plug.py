"""Original hidden-prong benchmark geometry and control constants.

Ported from the user-supplied legacy insertion task. All exterior, inertial,
and control parameters are shared; only the two prong meshes encode the variant.
"""

from functools import lru_cache
from itertools import product
from pathlib import Path

VARIANTS = ("xm", "xp", "ym", "yp")
OFFSETS = {"xm": (-0.015, 0.0), "xp": (0.015, 0.0), "ym": (0.0, -0.015), "yp": (0.0, 0.015)}
BODY_HALF = (0.032, 0.021, 0.015)
PRONG_HALF = (0.004, 0.004, 0.006)
HALF_SPACING = 0.011
INSERT_Z = 0.016
SUCCESS_DISTANCE = 0.002
SPAWN_CENTER = (0.0, 0.025)
SPAWN_HALF = (0.05, 0.05)
SPAWN_Z = 0.035
GRIP_OFFSET = (0.0, -0.0015, 0.045)
ACTION_LOW = (-0.07, -0.045, 0.055)
ACTION_HIGH = (0.07, 0.076, 0.09)
TCP_MAX_DELTA = 0.03
CAMERA_BASE = (-0.02, 0.50, 0.05)
CAMERA_BASE_YAW = -1.494
GIMBAL_HOME = (0.0, 0.19, 0.22, -1.42, 0.837)
GIMBAL_RANGE = (0.05, 0.05, 0.045, 0.42, 0.27)
GIMBAL_DELTA = (0.01, 0.01, 0.01, 0.05, 0.05)
HOLDER = Path(__file__).parent / "assets/insertion/holder1.stl"


def variant_assignment(n, variant=None):
    """Fixed balanced per-world assignment, matching the legacy VariantEntityCfg."""
    if variant is not None:
        return [VARIANTS.index(variant)] * n
    return [i % len(VARIANTS) for i in range(n)]


def _raw_spec(dx, dy):
    import mujoco

    spec = mujoco.MjSpec()
    spec.add_mesh(name="holder1", file=str(HOLDER), scale=(0.001, 0.001, 0.001))
    spec.add_material(name="plastic", rgba=(0.92, 0.92, 0.95, 1), specular=0.2, shininess=0.18)
    spec.add_material(name="brass", rgba=(0.72, 0.55, 0.25, 1), specular=0.35, shininess=0.45)
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="free")
    body.add_geom(
        name="housing",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0, 0, BODY_HALF[2]),
        size=BODY_HALF,
        material="plastic",
        density=250,
        friction=(0.8, 0.005, 0.0001),
        group=0,
    )
    for tag, sign in (("a", 1), ("b", -1)):
        center = (dx + sign * HALF_SPACING, dy, -PRONG_HALF[2])
        vertices = [
            c + s * h
            for signs in product((-1, 1), repeat=3)
            for c, s, h in zip(center, signs, PRONG_HALF, strict=True)
        ]
        # Primitive positions cannot vary per world in MjLab. Mesh vertices can.
        spec.add_mesh(name=f"prong_{tag}_mesh", uservert=vertices)
        body.add_geom(
            name=f"prong_{tag}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"prong_{tag}_mesh",
            material="brass",
            density=250,
            friction=(0.8, 0.005, 0.0001),
            group=0,
        )
    body.add_geom(
        name="collar",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="holder1",
        material="plastic",
        friction=(0.5, 0.005, 0.0001),
        group=0,
    )
    # Identical diagnostic site across variants. Offset correction is privileged
    # task state, never a site-position side channel in the actor observation.
    body.add_site(name="tip", pos=(0, 0, -0.012), size=(0.001,) * 3, group=5)
    return spec


@lru_cache
def reference_inertial():
    body = _raw_spec(0, 0).compile().body("object")
    return float(body.mass[0]), body.ipos.copy(), body.iquat.copy(), body.inertia.copy()


def object_spec(variant="xm"):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown plug variant: {variant}")
    spec = _raw_spec(*OFFSETS[variant])
    body = spec.body("object")
    body.explicitinertial = True
    body.mass, body.ipos, body.iquat, body.inertia = reference_inertial()
    return spec


def fixture_spec(clearance=0.002):
    import mujoco

    spec = mujoco.MjSpec()
    spec.add_material(
        name="socket_finish", rgba=(0.62, 0.66, 0.70, 1), specular=0.15, shininess=0.2
    )
    body = spec.worldbody.add_body(name="fixture", mocap=True)
    px, py, pz = 0.05, 0.035, 0.007
    h, s = PRONG_HALF[0] + clearance, HALF_SPACING
    yh, xh = (py - h) / 2, (px - s - h) / 2
    for name, pos, size in (
        ("front", (0, -h - yh, pz), (px, yh, pz)),
        ("back", (0, h + yh, pz), (px, yh, pz)),
        ("left", (-s - h - xh, 0, pz), (xh, h, pz)),
        ("mid", (0, 0, pz), (s - h, h, pz)),
        ("right", (s + h + xh, 0, pz), (xh, h, pz)),
    ):
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            material="socket_finish",
            friction=(0.3, 0.005, 0.0001),
            priority=1,
            solref=(0.007, 1),
            group=0,
        )
    body.add_site(name="goal", pos=(0, 0, INSERT_Z), size=(0.002,) * 3, group=5)
    return spec
