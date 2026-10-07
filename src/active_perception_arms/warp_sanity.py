"""Run scripted controls through actual MjLab/Warp physics and RGB sensors."""

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from .config import OCCLUSIONS, TASKS, Experiment
from .environment import make_env_cfg
from .mdp import instantaneous_success
from .native import NativeEnv, ScriptedPolicy
from .sanity import OverviewRecorder, mosaic
from .scenes import OVERVIEW_SIZE


def sync_native_state(proxy, env, step, index=0):
    """Mirror Warp state for diagnostic IK and the OpenGL observer camera."""
    proxy.data.qpos[:] = env.sim.data.qpos[index].cpu().numpy()
    proxy.data.qvel[:] = env.sim.data.qvel[index].cpu().numpy()
    origin = env.scene.env_origins[index].cpu().numpy()
    proxy.data.qpos[proxy.object_adr : proxy.object_adr + 3] -= origin
    for body in ("fixture/fixture", "occluder/panel"):
        native_id = proxy.model.body(body).mocapid[0]
        warp_id = env.sim.mj_model.body(body).mocapid[0]
        proxy.data.mocap_pos[native_id] = (
            env.sim.data.mocap_pos[index, warp_id].cpu().numpy() - origin
        )
        proxy.data.mocap_quat[native_id] = env.sim.data.mocap_quat[index, warp_id].cpu().numpy()
    proxy.data.time = step * proxy.cfg.step_dt
    proxy.step_count = step
    for name in proxy.targets:
        proxy.targets[name] = (
            env.action_manager.get_term("arms").targets[name][index].cpu().numpy().copy()
        )
    if proxy.cfg.task == "plug":
        term = env.action_manager.get_term("arms")
        proxy.tcp_target = term.tcp_target[index].cpu().numpy().copy()
        proxy.gimbal_target = term.gimbal_target[index].cpu().numpy().copy()
    mujoco.mj_forward(proxy.model, proxy.data)


def run(task, output, device="cpu", seconds=8.0, occlusion="phase", overview=True):
    cfg = Experiment(
        task=task, num_envs=1, randomize=False, occlusion=occlusion, episode_seconds=seconds
    )
    setup = make_env_cfg(cfg)
    # Diagnostic rollouts continue after success so terminal poses remain visible.
    setup.terminations = {}
    env = ManagerBasedRlEnv(setup, device=device)
    proxy = NativeEnv(cfg)
    if env.sim.mj_model.nq != proxy.model.nq:
        raise RuntimeError("Diagnostic proxy topology differs from training scene")
    for joint in range(proxy.model.njnt):
        # Match every address explicitly before copying coordinates between backends.
        name = mujoco.mj_id2name(proxy.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if not np.array_equal(
            proxy.model.joint(name).qposadr, env.sim.mj_model.joint(name).qposadr
        ):
            raise RuntimeError("Joint ordering differs between native and Warp scene")
    policy = ScriptedPolicy(proxy)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    observer = (
        OverviewRecorder(
            proxy.model, output / f"{task}_warp_{occlusion}_overview.mp4", fps=1 / cfg.step_dt
        )
        if overview
        else None
    )
    frames = []
    successful, hold = False, 0
    try:
        obs, _ = env.reset()
        sync_native_state(proxy, env, 0)
        with imageio.get_writer(
            output / f"{task}_warp_{occlusion}.mp4", fps=1 / (cfg.step_dt * 3), macro_block_size=2
        ) as writer:
            for step in range(math.ceil(seconds / cfg.step_dt)):
                # Diagnostic-only CPU synchronization for analytical IK. The learned
                # joint-action environment never copies images or state to the CPU.
                action = torch.as_tensor(policy(), device=device, dtype=torch.float32)[None]
                obs, _, _, _, _ = env.step(action)
                sync_native_state(proxy, env, step + 1)
                if observer:
                    observer.capture(proxy.data)

                hold = hold + 1 if instantaneous_success(env)[0].item() else 0
                successful = successful or hold >= 3
                if step % 3 == 0:
                    images = [obs[name][0].permute(1, 2, 0).cpu().numpy() for name in cfg.sensors]
                    tile = mosaic(
                        images,
                        [
                            f"{name}: Warp RGB, t={(step + 1) * cfg.step_dt:.2f}s"
                            for name in cfg.sensors
                        ],
                    )
                    writer.append_data(np.asarray(tile))
                    if step % 60 == 0:
                        frames.append(tile)
        from PIL import Image

        gallery = Image.new("RGB", (frames[0].width, sum(f.height for f in frames)))
        for index, frame in enumerate(frames):
            gallery.paste(frame, (0, index * frame.height))
        gallery.save(output / f"{task}_warp_{occlusion}.png")
        if observer:
            observer.save_frame(output / f"{task}_warp_{occlusion}_overview.png")
        report = {
            "task": task,
            "backend": "MjLab / MuJoCo Warp",
            "device": device,
            "occlusion": occlusion,
            "scripted_success": successful,
            "final_success": bool(hold >= 3),
            "training_started": False,
            "resolution": [cfg.width, cfg.height],
            "overview_resolution": list(OVERVIEW_SIZE) if observer else None,
            "overview_renderer": "native MuJoCo OpenGL of Warp state" if observer else None,
        }
        (output / f"{task}_warp_{occlusion}.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        if observer:
            observer.close()
        env.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--occlusion", choices=OCCLUSIONS, default="phase")
    p.add_argument("--no-overview", action="store_true", help="Skip the 1080p observer render")
    p.add_argument("--output", type=Path, default=Path("artifacts/warp_sanity"))
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    for task in args.tasks:
        print(
            json.dumps(
                run(
                    task,
                    args.output,
                    args.device,
                    args.seconds,
                    args.occlusion,
                    overview=not args.no_overview,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
