"""Restore-task diagnostics: four-variant videos, matched views, actual Warp RGB.

No learner is constructed. The insertion controller uses privileged geometry;
camera motion is a fixed schedule and is not an adaptive visual policy.
"""

import argparse
import json
import math
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image

from . import mdp, plug
from .config import Experiment, static_candidates
from .environment import make_env_cfg
from .native import NativeEnv, ScriptedPolicy
from .sanity import CameraAudit, OverviewRecorder, mosaic, run
from .scenes import look_at_quat
from .warp_sanity import sync_native_state


def matched_views(output):
    """Matched reset states; segmentation measures geometry, not policy success."""
    output.mkdir(parents=True, exist_ok=True)
    rows, images = {}, {}
    grid = static_candidates("plug")
    for variant in plug.VARIANTS:
        env = NativeEnv(Experiment(plug_variant=variant, num_envs=1, randomize=False))
        audit = CameraAudit(env)
        views = {}
        try:
            for name in ("manipulator/wrist_cam", "camera_arm/wrist_cam"):
                rgb, metrics = audit.render(name)
                views[name] = metrics
                images.setdefault(name, []).append(rgb)
            for index, position in enumerate(grid):
                cam = env.model.camera("fixed")
                cam.pos = position
                cam.quat = look_at_quat(position, env.cfg.fixed_lookat)
                mujoco.mj_forward(env.model, env.data)
                rgb, metrics = audit.render("fixed")
                key = f"fixed_{index:02d}"
                views[key] = metrics
                images.setdefault(key, []).append(rgb)
            rows[variant] = views
        finally:
            audit.close()
    differences = {}
    for view, frames in images.items():
        differences[view] = {
            f"{plug.VARIANTS[i]}:{plug.VARIANTS[j]}": int(
                np.any(frames[i] != frames[j], axis=-1).sum()
            )
            for i in range(4)
            for j in range(i + 1, 4)
        }
    # A visibility diagnostic only, not validation-success-based view selection.
    most_distinct = max(range(len(grid)), key=lambda i: min(differences[f"fixed_{i:02d}"].values()))
    selected = ("manipulator/wrist_cam", "camera_arm/wrist_cam", f"fixed_{most_distinct:02d}")
    tiles = [mosaic(images[name], [f"{v}: {name}" for v in plug.VARIANTS]) for name in selected]
    gallery = Image.new("RGB", (tiles[0].width, sum(tile.height for tile in tiles)))
    for i, tile in enumerate(tiles):
        gallery.paste(tile, (0, i * tile.height))
    gallery.save(output / "matched_reset_views.png")
    report = {
        "task_revision": "hidden_prongs_v1",
        "policy_resolution": [128, 96],
        "state": "Same deterministic reset, no dynamics; shadows disabled",
        "scope": "Visibility diagnostic only. Does not select the best task-success baseline.",
        "views": rows,
        "pairwise_differing_rgb_pixels": differences,
        "gallery_fixed_view": most_distinct,
        "fixed_grid": grid,
    }
    (output / "matched_views.json").write_text(json.dumps(report, indent=2))
    return {"wrist_pairwise_pixels": differences[selected[0]], "gallery_fixed_view": most_distinct}


def warp_rollout(output, device="cpu", seconds=3.5):
    from mjlab.envs import ManagerBasedRlEnv

    output.mkdir(parents=True, exist_ok=True)
    cfg = Experiment(num_envs=4, randomize=False, episode_seconds=seconds)
    setup = make_env_cfg(cfg)
    setup.terminations = {}
    env = ManagerBasedRlEnv(setup, device=device)
    proxies = [NativeEnv(replace(cfg, num_envs=1, plug_variant=v)) for v in plug.VARIANTS]
    policies = [ScriptedPolicy(proxy) for proxy in proxies]
    won = torch.zeros(4, device=device, dtype=torch.bool)
    hold = torch.zeros(4, device=device, dtype=torch.long)
    records, snapshots = [], []
    try:
        with ExitStack() as stack:
            writer = stack.enter_context(
                imageio.get_writer(
                    output / "plug_all_variants_warp.mp4",
                    fps=1 / (cfg.step_dt * 3),
                    macro_block_size=2,
                )
            )
            observers = []
            for variant, proxy in zip(plug.VARIANTS, proxies, strict=True):
                observer = OverviewRecorder(
                    proxy.model,
                    output / f"plug_{variant}_warp_detail.mp4",
                    fps=1 / cfg.step_dt,
                    camera="task_detail",
                )
                observers.append(observer)
                stack.callback(observer.close)
            obs, _ = env.reset()
            # At matched reset, test both actual Warp camera outputs for hidden-state cues.
            reset_differences = {
                name: {
                    f"{plug.VARIANTS[i]}:{plug.VARIANTS[j]}": int(
                        (obs[name][i] != obs[name][j]).any(0).sum().item()
                    )
                    for i in range(4)
                    for j in range(i + 1, 4)
                }
                for name in cfg.sensors
            }
            for step in range(math.ceil(seconds / cfg.step_dt)):
                for i, proxy in enumerate(proxies):
                    sync_native_state(proxy, env, step, i)
                action = torch.as_tensor(
                    np.stack([p() for p in policies]), device=device, dtype=torch.float32
                )
                obs, _, _, _, _ = env.step(action)
                if not torch.isfinite(env.sim.data.qpos).all():
                    raise RuntimeError("Non-finite Warp physics")
                hold = torch.where(mdp.instantaneous_success(env), hold + 1, 0)
                won |= hold >= 3
                for i, observer in enumerate(observers):
                    sync_native_state(proxies[i], env, step + 1, i)
                    observer.capture(proxies[i].data)
                if step % 3 == 0:
                    tiles = []
                    for i, variant in enumerate(plug.VARIANTS):
                        tiles.append(
                            mosaic(
                                [
                                    obs[name][i].permute(1, 2, 0).cpu().numpy()
                                    for name in cfg.sensors
                                ],
                                [
                                    f"{variant} {name} | Warp RGB | {(step + 1) * cfg.step_dt:.2f}s"
                                    for name in cfg.sensors
                                ],
                                scale=2,
                            )
                        )
                    tile = Image.new("RGB", (tiles[0].width * 2, tiles[0].height * 2))
                    for i, view in enumerate(tiles):
                        tile.paste(view, ((i % 2) * view.width, (i // 2) * view.height))
                    writer.append_data(np.asarray(tile))
                    if step in (0, 30, 60):
                        snapshots.append(tile)
                    records.append(
                        {"time": (step + 1) * cfg.step_dt, "success": (hold >= 3).cpu().tolist()}
                    )
            for variant, observer in zip(plug.VARIANTS, observers, strict=True):
                observer.save_frame(output / f"plug_{variant}_warp_detail.png")
            gallery = Image.new("RGB", (snapshots[0].width, sum(f.height for f in snapshots)))
            for i, frame in enumerate(snapshots):
                gallery.paste(frame, (0, i * frame.height))
            gallery.save(output / "plug_all_variants_warp.png")
            _, obj, _, goal = mdp.positions(env)
            report = {
                "backend": "MjLab / MuJoCo Warp",
                "device": device,
                "task_revision": cfg.task_revision,
                "variants": plug.VARIANTS,
                "scripted_success": won.cpu().tolist(),
                "final_success": (hold >= 3).cpu().tolist(),
                "final_position_error_m": torch.linalg.vector_norm(obj - goal, dim=-1)
                .cpu()
                .tolist(),
                "reset_pairwise_differing_rgb_pixels": reset_differences,
                "controller": "Privileged scripted manipulation; blind scheduled camera motion",
                "resolution": [cfg.width, cfg.height],
                "training_started": False,
                "detail_renderer": "1920x1080 native OpenGL of actual per-variant Warp state",
                "frames": records,
            }
            (output / "warp_report.json").write_text(json.dumps(report, indent=2))
            return {k: v for k, v in report.items() if k != "frames"}
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "warp", "both"), default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=3.5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/plug_restoration"))
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.backend in ("native", "both"):
        print(json.dumps(matched_views(args.output / "visibility")), flush=True)
        for variant in plug.VARIANTS:
            print(
                json.dumps(
                    run("plug", args.output / "native", seconds=args.seconds, plug_variant=variant)
                ),
                flush=True,
            )
    if args.backend in ("warp", "both"):
        print(json.dumps(warp_rollout(args.output / "warp", args.device, args.seconds)), flush=True)


if __name__ == "__main__":
    main()
