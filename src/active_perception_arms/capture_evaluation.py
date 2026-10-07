"""Save representative trajectories during evaluation; render without resimulating."""

import json
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from . import mdp, plug
from .config import saved_experiment
from .native import NativeEnv
from .sanity import OverviewRecorder, mosaic


class RepresentativeCapture:
    """Buffer one evaluation batch on host, retain first success/failure per variant."""

    def __init__(self, directory, cfg):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.selected = set()
        self.records = []
        self.frames = []

    def step(self, env, obs, action):
        _, obj, _, goal = mdp.positions(env)
        fixture = env.sim.mj_model.body("fixture/fixture").mocapid[0]
        panel = env.sim.mj_model.body("occluder/panel").mocapid[0]
        self.frames.append(
            {
                "qpos": env.sim.data.qpos.cpu().numpy().copy(),
                "qvel": env.sim.data.qvel.cpu().numpy().copy(),
                "fixture": env.sim.data.mocap_pos[:, fixture].cpu().numpy().copy(),
                "panel": env.sim.data.mocap_pos[:, panel].cpu().numpy().copy(),
                "origins": env.scene.env_origins.cpu().numpy().copy(),
                "action": action.cpu().numpy().copy(),
                "error": (obj - goal).norm(dim=-1).cpu().numpy().copy(),
                "camera_targets": env.action_manager.get_term("arms")
                .targets["camera_arm"]
                .cpu()
                .numpy()
                .copy(),
                **{name: obs[name].cpu().numpy().copy() for name in self.cfg.sensors},
            }
        )

    def reward(self, reward):
        self.frames[-1]["reward"] = reward.cpu().numpy().copy()

    def finish_batch(self, batch, won, elapsed, count):
        variants = plug.variant_assignment(self.cfg.num_envs, self.cfg.plug_variant)
        for i in range(count):
            variant = plug.VARIANTS[variants[i]]
            success = bool(won[i].item())
            key = (variant, success)
            if key in self.selected:
                continue
            self.selected.add(key)
            episode = batch * self.cfg.num_envs + i
            length = min(len(self.frames), round(elapsed[i].item() / self.cfg.step_dt))
            path = (
                self.directory
                / f"{variant}_{'success' if success else 'failure'}_episode{episode}.npz"
            )
            arrays = {
                name: np.stack([f[name][i] for f in self.frames[:length]])
                for name in self.frames[0]
            }
            np.savez_compressed(path, **arrays)
            self.records.append(
                dict(
                    episode=episode,
                    variant=variant,
                    expected_success=success,
                    reproduced_success=success,
                    source="captured from the evaluated trajectory, no physics replay",
                    trace=str(path),
                    steps=[
                        dict(
                            time=j * self.cfg.step_dt,
                            position_error_m=float(arrays["error"][j]),
                            action=arrays["action"][j].tolist(),
                            reward=float(arrays["reward"][j]),
                        )
                        for j in range(length)
                    ],
                )
            )
        self.frames = []
        path = self.directory / "capture.json"
        path.write_text(
            json.dumps(dict(experiment=self.cfg.to_dict(), records=self.records), indent=2)
        )
        return str(path)


def render_capture(report, output):
    capture = json.loads(Path(report["representative_capture"]).read_text())
    cfg = saved_experiment(capture["experiment"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = capture["records"]
    for record in records:
        name = Path(record["trace"]).stem
        proxy = NativeEnv(replace(cfg, num_envs=1, plug_variant=record["variant"]))
        outside = output / f"{name}_outside.mp4"
        policy = output / f"{name}_policy.mp4"
        observer = OverviewRecorder(proxy.model, outside, fps=1 / cfg.step_dt)
        try:
            with (
                np.load(record["trace"]) as data,
                imageio.get_writer(policy, fps=1 / cfg.step_dt, macro_block_size=2) as writer,
            ):
                for step in range(len(data["qpos"])):
                    origin = data["origins"][step]
                    proxy.data.qpos[:] = data["qpos"][step]
                    proxy.data.qvel[:] = data["qvel"][step]
                    proxy.data.qpos[proxy.object_adr : proxy.object_adr + 3] -= origin
                    for body, key in [("fixture/fixture", "fixture"), ("occluder/panel", "panel")]:
                        proxy.data.mocap_pos[proxy.model.body(body).mocapid[0]] = (
                            data[key][step] - origin
                        )
                    mujoco.mj_forward(proxy.model, proxy.data)
                    observer.capture(proxy.data)
                    frames = [data[name][step].transpose(1, 2, 0) for name in cfg.sensors]
                    tile = mosaic(
                        frames,
                        [
                            f"{name}: captured input t={step * cfg.step_dt:.2f}s"
                            for name in cfg.sensors
                        ],
                    )
                    writer.append_data(np.asarray(tile))
                if cfg.condition == "initial":
                    targets = data["camera_targets"][round(cfg.initial_seconds / cfg.step_dt) :]
                    record["camera_target_max_change_after_inspection"] = (
                        float(np.max(np.abs(targets - targets[:1]))) if len(targets) else None
                    )
        finally:
            observer.close()
        record.update(outside_video=str(outside), policy_video=str(policy))
    result = dict(
        selection="First observed success/failure per variant in recorded evaluation repeat",
        source="Captured actual trajectory and actor inputs; no second physics rollout",
        records=records,
    )
    (output / "representatives.json").write_text(json.dumps(result, indent=2))
    return records
