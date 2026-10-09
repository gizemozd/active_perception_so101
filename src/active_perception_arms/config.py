"""Experiment definitions shared by native diagnostics and GPU environments."""

import math
from dataclasses import asdict, dataclass
from typing import Literal

Task = Literal["plug", "transfer", "push"]
Condition = Literal["wrist", "static", "wrist_static", "initial", "scheduled", "active"]
Occlusion = Literal["clean", "static", "phase", "random", "dynamic"]
TASKS = ("plug", "transfer", "push")
CONDITIONS = ("wrist", "static", "wrist_static", "initial", "scheduled", "active")
OCCLUSIONS = ("clean", "static", "phase", "random", "dynamic")
SUCCESS_STATE_SAMPLES = ("derived_substep", "current_qpos")
REWARD_PROFILES = ("progress", "legacy_log_hold")
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
    occlusion: Occlusion | None = None
    seed: int = 0
    num_envs: int = 256
    width: int | None = None
    height: int | None = None
    timestep: float = 0.002
    decimation: int = 20
    episode_seconds: float | None = None
    initial_seconds: float = 1.0
    joint_step: float = 0.035
    randomize: bool = True
    perturb_push: bool = False
    fixed_position: tuple[float, float, float] | None = None
    fixed_lookat: tuple[float, float, float] | None = None
    clearance: float = 0.002
    render_sensors: bool = True
    memory: Literal["gru", "none"] = "gru"
    plug_variant: Literal["xm", "xp", "ym", "yp"] | None = None
    task_revision: str | None = None
    # Derived positions lag integration by one physics substep in MjLab1.4.
    # Existing saved runs retain their original sampling in saved_experiment().
    success_state_sample: Literal["derived_substep", "current_qpos"] = "current_qpos"
    reward_profile: Literal["progress", "legacy_log_hold"] = "progress"
    # Opt-in optical sweep. None for every legacy condition/checkpoint; its
    # existing scene, random draws, timer and observation dimensions stay intact.
    occlusion_revision: str | None = None
    dynamic_onset_range: tuple[float, float] | None = None
    dynamic_duration_range: tuple[float, float] | None = None
    dynamic_center: tuple[float, float, float] | None = None
    dynamic_center_jitter: tuple[float, float, float] | None = None
    dynamic_travel_range: tuple[float, float] | None = None
    dynamic_panel_half_size: tuple[float, float, float] | None = None
    dynamic_panel_yaw: float | None = None

    def __post_init__(self):
        from . import plug

        defaults = {
            "task_revision": "hidden_prongs_v1" if self.task == "plug" else "v1",
            "occlusion": "clean" if self.task == "plug" else "random",
            "width": 128 if self.task == "plug" else 96,
            "height": 96 if self.task == "plug" else 72,
            # Same inspection pause for every sensing condition, followed by the
            # original plug task's 2.5-second manipulation budget.
            "episode_seconds": self.initial_seconds + 2.5 if self.task == "plug" else 12.0,
            "fixed_position": plug.GIMBAL_HOME[:3] if self.task == "plug" else CAMERA_HOME,
            "fixed_lookat": (0.022, 0.045, 0.047) if self.task == "plug" else CAMERA_LOOKAT,
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)
        if self.task == "plug" and self.task_revision != "hidden_prongs_v1":
            raise ValueError(
                "The centered-pin plug prototype is incompatible with hidden_prongs_v1"
            )
        for value, choices, name in (
            (self.task, TASKS, "task"),
            (self.condition, CONDITIONS, "condition"),
            (self.occlusion, OCCLUSIONS, "occlusion"),
            (self.success_state_sample, SUCCESS_STATE_SAMPLES, "success_state_sample"),
            (self.reward_profile, REWARD_PROFILES, "reward_profile"),
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
        if self.plug_variant is not None and (
            self.task != "plug" or self.plug_variant not in plug.VARIANTS
        ):
            raise ValueError("plug_variant requires plug and one of xm/xp/ym/yp")
        if self.reward_profile == "legacy_log_hold" and (
            self.task != "plug" or self.success_state_sample != "current_qpos"
        ):
            raise ValueError("legacy_log_hold requires plug and current_qpos success sampling")
        self._validate_dynamic_occlusion()

    def _validate_dynamic_occlusion(self):
        fields = (
            "occlusion_revision",
            "dynamic_onset_range",
            "dynamic_duration_range",
            "dynamic_center",
            "dynamic_center_jitter",
            "dynamic_travel_range",
            "dynamic_panel_half_size",
            "dynamic_panel_yaw",
        )
        if self.occlusion != "dynamic":
            if any(getattr(self, name) is not None for name in fields):
                raise ValueError("Dynamic occlusion parameters require occlusion=dynamic")
            return
        # The sweep clears before timeout. Waiting remains an available strategy,
        # and plug timing fits its 3.5-second horizon without extending its budget.
        horizon = self.episode_seconds
        defaults = {
            "occlusion_revision": "world_sweep_v1",
            "dynamic_onset_range": tuple(
                x * horizon for x in ((0.10, 0.42) if self.task == "plug" else (1 / 6, 5 / 12))
            ),
            "dynamic_duration_range": tuple(
                x * horizon for x in ((0.16, 0.32) if self.task == "plug" else (1 / 12, 7 / 24))
            ),
            "dynamic_center": (0.0, 0.095, 0.105)
            if self.task == "plug"
            else (-0.045, -0.105, 0.155),
            "dynamic_center_jitter": (0.015, 0.010, 0.010),
            "dynamic_travel_range": (0.10, 0.14),
            "dynamic_panel_half_size": (0.040, 0.004, 0.070),
            "dynamic_panel_yaw": 0.0,
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)
        if self.occlusion_revision != "world_sweep_v1":
            raise ValueError("Unsupported dynamic occlusion revision")
        for name in fields[1:-1]:
            value = tuple(getattr(self, name))
            expected = 2 if name.endswith("range") else 3
            if len(value) != expected or not all(math.isfinite(x) for x in value):
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, value)
        onset, duration, travel = (
            self.dynamic_onset_range,
            self.dynamic_duration_range,
            self.dynamic_travel_range,
        )
        if not 0 <= onset[0] <= onset[1] < horizon:
            raise ValueError("dynamic_onset_range must fall inside the episode")
        if not 0 < duration[0] <= duration[1]:
            raise ValueError("dynamic_duration_range must be positive and ordered")
        if onset[1] + duration[1] > horizon - self.step_dt:
            raise ValueError("Dynamic panel must clear before the episode ends")
        if not 0 < travel[0] <= travel[1]:
            raise ValueError("dynamic_travel_range must be positive and ordered")
        if any(x < 0 for x in self.dynamic_center_jitter):
            raise ValueError("dynamic_center_jitter must be nonnegative")
        if any(x <= 0 for x in self.dynamic_panel_half_size):
            raise ValueError("dynamic_panel_half_size must be positive")
        if not math.isfinite(self.dynamic_panel_yaw):
            raise ValueError("dynamic_panel_yaw must be finite")

    @property
    def step_dt(self):
        return self.timestep * self.decimation

    @property
    def camera_control(self):
        return self.condition in ("initial", "active")

    @property
    def manip_dim(self):
        return {"plug": 3, "transfer": 6, "push": 5}[self.task]

    @property
    def camera_base(self):
        from . import plug

        return plug.CAMERA_BASE if self.task == "plug" else CAMERA_BASE

    @property
    def camera_base_yaw(self):
        from . import plug

        return plug.CAMERA_BASE_YAW if self.task == "plug" else CAMERA_BASE_YAW

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


def static_candidates(task=None):
    """Broad view search including overhead; no test-set selection happens here."""
    import math

    from . import plug

    poses = [plug.GIMBAL_HOME[:3] if task == "plug" else CAMERA_HOME, (0.0, 0.025, 0.60)]
    # Include grazing views of the underside: overhead-only searches would make
    # the hidden-prong fixed-camera comparison artificially weak.
    for height in (0.08, 0.22, 0.35) if task == "plug" else (0.22, 0.35, 0.50):
        for azimuth in (-135, -90, -45, 0, 45, 90, 135, 180):
            a = math.radians(azimuth)
            radius = 0.18 if height == 0.08 else 0.30
            poses.append((radius * math.cos(a), 0.025 + radius * math.sin(a), height))
    return poses


def saved_experiment(data):
    """Reject old plug checkpoints instead of silently changing their task."""
    if data.get("task", "plug") == "plug" and data.get("task_revision") != "hidden_prongs_v1":
        raise ValueError("Legacy centered-pin experiment; restore its code revision to evaluate it")
    return Experiment(
        **{"success_state_sample": "derived_substep", "reward_profile": "progress", **data}
    )
