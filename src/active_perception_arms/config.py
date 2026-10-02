"""Experiment definitions shared by native diagnostics and GPU environments."""

from dataclasses import asdict, dataclass
from typing import Literal

Task = Literal["plug", "transfer", "push"]
Condition = Literal["wrist", "static", "wrist_static", "initial", "scheduled", "active"]
Occlusion = Literal["clean", "static", "phase", "random"]
TASKS = ("plug", "transfer", "push")
CONDITIONS = ("wrist", "static", "wrist_static", "initial", "scheduled", "active")
OCCLUSIONS = ("clean", "static", "phase", "random")
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JOINTS = ARM_JOINTS + ("gripper",)
MANIP_BASE = (-0.19, 0.025, 0.05)
CAMERA_BASE = (-0.30, -0.44, 0.05)
CAMERA_BASE_YAW = 0.950
CAMERA_HOME = (-0.119, -0.187, 0.274)
CAMERA_LOOKAT = (0.0, 0.025, 0.045)
STOW = (0.0, -0.4, 0.9, 0.6, 0.0, 0.6)
SOURCE_XY = (-0.025, -0.045)
TARGET_XY = (0.025, 0.055)
FOVY = 48.4554906359


@dataclass(frozen=True)
class Experiment:
    task: Task = "plug"
    condition: Condition = "active"
    occlusion: Occlusion = "random"
    seed: int = 0
    num_envs: int = 256
    width: int = 96
    height: int = 72
    timestep: float = 0.002
    decimation: int = 20
    episode_seconds: float = 12.0
    initial_seconds: float = 1.0
    joint_step: float = 0.035
    randomize: bool = True
    perturb_push: bool = False
    fixed_position: tuple[float, float, float] = CAMERA_HOME
    fixed_lookat: tuple[float, float, float] = CAMERA_LOOKAT
    clearance: float = 0.002
    render_sensors: bool = True
    memory: Literal["gru", "none"] = "gru"

    def __post_init__(self):
        for value, choices, name in (
            (self.task, TASKS, "task"),
            (self.condition, CONDITIONS, "condition"),
            (self.occlusion, OCCLUSIONS, "occlusion"),
        ):
            if value not in choices:
                raise ValueError(f"Unknown {name}: {value}")
        if self.num_envs < 1 or self.width < 32 or self.height < 32:
            raise ValueError("num_envs must be positive and images at least 32 pixels per side")
        if self.timestep <= 0 or self.decimation < 1 or self.joint_step <= 0:
            raise ValueError("Invalid control timing or action scale")
        if not 0 <= self.initial_seconds < self.episode_seconds:
            raise ValueError("initial_seconds must be inside the episode")
        if not 0.0005 <= self.clearance <= 0.005:
            raise ValueError("clearance must be between 0.5 and 5 mm")
        if self.perturb_push and self.task != "push":
            raise ValueError("perturb_push is only defined for pushing")
        if self.memory not in ("gru", "none"):
            raise ValueError("memory must be gru or none")

    @property
    def step_dt(self):
        return self.timestep * self.decimation

    @property
    def camera_control(self):
        return self.condition in ("initial", "active")

    @property
    def manip_dim(self):
        return 6 if self.task == "transfer" else 5

    @property
    def action_dim(self):
        return self.manip_dim + (5 if self.camera_control else 0)

    @property
    def sensors(self):
        if self.condition == "wrist":
            return ("wrist",)
        if self.condition == "static":
            return ("external",)
        return ("wrist", "external")

    def to_dict(self):
        return asdict(self)


def static_candidates():
    """Broad view search including overhead; no test-set selection happens here."""
    import math

    poses = [CAMERA_HOME, (0.0, 0.025, 0.60)]
    for height in (0.22, 0.35, 0.50):
        for azimuth in (-135, -90, -45, 0, 45, 90, 135, 180):
            a = math.radians(azimuth)
            poses.append((0.30 * math.cos(a), 0.025 + 0.30 * math.sin(a), height))
    return poses
