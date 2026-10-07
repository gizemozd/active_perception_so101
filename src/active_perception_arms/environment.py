"""MjLab manager-based environments backed directly by MuJoCo Warp."""

from functools import partial

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg, VariantEntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from . import mdp, plug, scenes
from .config import JOINTS, MANIP_BASE, Experiment
from .native import calibration


def arm_cfg(position, yaw, home):
    return EntityCfg(
        spec_fn=scenes.arm_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=JOINTS,
                    transmission_type=TransmissionType.JOINT,
                    command_field="position",
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=position, rot=scenes.yaw_quat(yaw), joint_pos=dict(zip(JOINTS, home, strict=True))
        ),
    )


def finish_mjlab_scene(spec, cfg):
    scenes.finish_scene(spec, cfg)
    if cfg.task == "plug":
        # MjLab 1.4 builds variants from a COPY of this spec. Entity indexing
        # still reads IDs from the original attached specs, which otherwise
        # remain -1 and silently address the last body/mocap/geom. Compile the
        # original once to populate IDs before Simulation initializes entities.
        spec.compile()


def make_env_cfg(cfg: Experiment):
    home = calibration(cfg.task)
    entities = {
        "manipulator": arm_cfg(MANIP_BASE, 0, home["manipulator"]),
        "camera_arm": arm_cfg(cfg.camera_base, cfg.camera_base_yaw, home["camera_arm"]),
        "object": EntityCfg(
            spec_fn=partial(scenes.object_spec, cfg.task),
            init_state=EntityCfg.InitialStateCfg(pos=(0, 0, 0.06), joint_pos={}),
        ),
        "fixture": EntityCfg(
            spec_fn=partial(scenes.fixture_spec, cfg.task, cfg.clearance),
            init_state=EntityCfg.InitialStateCfg(joint_pos={}),
        ),
        "occluder": EntityCfg(
            spec_fn=scenes.occluder_spec,
            init_state=EntityCfg.InitialStateCfg(pos=(0, 0, -1), joint_pos={}),
        ),
        "table": EntityCfg(
            spec_fn=partial(scenes.table_spec, cfg.task),
            init_state=EntityCfg.InitialStateCfg(joint_pos={}),
        ),
    }
    if cfg.task == "plug":
        entities["object"] = VariantEntityCfg(
            variants={v: partial(plug.object_spec, v) for v in plug.VARIANTS},
            assignment=partial(plug.variant_assignment, variant=cfg.plug_variant),
            init_state=EntityCfg.InitialStateCfg(pos=(0, 0, plug.SPAWN_Z), joint_pos={}),
        )
    sensors = []
    obs = {
        "proprio": ObservationGroupCfg(
            {"state": ObservationTermCfg(func=mdp.proprio)}, concatenate_terms=True
        ),
        "critic": ObservationGroupCfg(
            {"state": ObservationTermCfg(func=mdp.critic_state)}, concatenate_terms=True
        ),
    }
    if cfg.render_sensors:
        for name in cfg.sensors:
            camera_name = (
                "manipulator/wrist_cam"
                if name == "wrist"
                else (
                    "fixed"
                    if cfg.condition in ("static", "wrist_static")
                    else "camera_arm/wrist_cam"
                )
            )
            # fixed camera is created by this sensor before the scene callback.
            kwargs = {"camera_name": camera_name}
            if camera_name == "fixed":
                kwargs = {
                    "pos": cfg.fixed_position,
                    "quat": tuple(scenes.look_at_quat(cfg.fixed_position, cfg.fixed_lookat)),
                    "fovy": scenes.FOVY,
                }
            sensors.append(
                CameraSensorCfg(
                    name=name,
                    width=cfg.width,
                    height=cfg.height,
                    data_types=("rgb",),
                    use_shadows=False,
                    use_textures=cfg.task == "plug",
                    enabled_geom_groups=(0, 1, 2),
                    **kwargs,
                )
            )
            obs[name] = ObservationGroupCfg(
                {"rgb": ObservationTermCfg(func=mdp.rgb, params={"sensor": name})},
                concatenate_terms=True,
            )
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities=entities,
            sensors=tuple(sensors),
            num_envs=cfg.num_envs,
            env_spacing=1.5,
            spec_fn=partial(finish_mjlab_scene, cfg=cfg),
        ),
        seed=cfg.seed,
        decimation=cfg.decimation,
        episode_length_s=cfg.episode_seconds,
        observations=obs,
        actions={"arms": mdp.ArmsActionCfg(entity_name="manipulator", experiment=cfg)},
        events={"reset": EventTermCfg(func=mdp.reset_task, mode="reset", params={"cfg": cfg})},
        rewards={"task": RewardTermCfg(func=mdp.task_reward, weight=1)},
        terminations={
            "success": TerminationTermCfg(func=mdp.success),
            "failure": TerminationTermCfg(func=mdp.failure),
            "timeout": TerminationTermCfg(func=time_out, time_out=True),
        },
        metrics={"success_rate": MetricsTermCfg(func=mdp.success_metric, reduce="last")},
        sim=SimulationCfg(
            nconmax=256,
            njmax=768,
            mujoco=MujocoCfg(
                timestep=cfg.timestep,
                iterations=30,
                ls_iterations=10,
                cone="pyramidal",
                integrator="implicitfast",
            ),
        ),
        scale_rewards_by_dt=False,
        is_finite_horizon=True,
    )


def make_env(cfg: Experiment, device="cuda:0"):
    return ManagerBasedRlEnv(make_env_cfg(cfg), device=device)
