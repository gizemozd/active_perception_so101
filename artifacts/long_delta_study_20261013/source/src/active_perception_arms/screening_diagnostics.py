"""Motion, alternative-strategy, and RGB diagnostics for the visibility prototypes.

No learning and no manipulation-success claim. Kinematic paths use the deployed
joint command increment; selected videos additionally execute native MuJoCo servo
dynamics. Scene/phase changes and the query marker remain prescribed interventions.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from .config import JOINTS
from .task_screening import CANDIDATES, GROUPS, ScreenScene


def sensing(scene):
    return any(
        scene.visibility(*scene.camera_view(name))["visible"]
        for name in ("camera_arm/wrist_cam", "manipulator/wrist_cam")
    )


def kinematic_path(scene, start, end):
    """Check every <=.035-rad command sample; finite samples are not swept volumes."""
    limits = np.array([scene.model.joint("camera_arm/" + j).range for j in JOINTS[:5]])
    if np.any(end < limits[:, 0]) or np.any(end > limits[:, 1]):
        return {"feasible": False, "reason": "joint_limit", "duration_s": 0.0, "first_seen_s": None}
    steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / 0.035)))
    first = None
    for step in range(steps + 1):
        scene.set_camera_arm(start + (end - start) * (step / steps))
        contacts = scene.camera_contacts()
        if contacts:
            return {
                "feasible": False,
                "reason": "collision",
                "contacts": contacts,
                "duration_s": step * 0.04,
                "first_seen_s": first,
            }
        if first is None and sensing(scene):
            first = step * 0.04
    return {"feasible": True, "duration_s": steps * 0.04, "first_seen_s": first}


def oracle_route(scene, catalog, start):
    order = sorted(
        range(len(catalog)), key=lambda i: np.max(np.abs(np.array(catalog[i]["q"]) - start))
    )
    for idx in order:
        q = np.array(catalog[idx]["q"])
        scene.set_camera_arm(q)
        if scene.camera_contacts() or not sensing(scene):
            continue
        route = kinematic_path(scene, start, q)
        if route["feasible"]:
            return {"index": idx, **route}, q
    scene.set_camera_arm(start)
    return {"index": None, "feasible": False, "duration_s": 0.0, "first_seen_s": None}, start


def controls(scene, seeds, catalog, fixed, scan_pair, scheduled_pair):
    records = []
    fixed_view = (np.array(fixed["position"]), np.array(fixed["matrix"]))
    for seed in seeds:
        oracle_start = scene.home["camera_arm"][:5].copy()
        scan_start = oracle_start.copy()
        for phase in (1, 2):
            # A phase schedule can stow before the hand relocates. This avoids
            # interpreting a prescribed phase reset into an arm as an RL benefit.
            scene.set_episode(seed, phase)
            home = scene.home["camera_arm"][:5].copy()
            scheduled_q = np.array(catalog[scheduled_pair[phase - 1]]["q"])
            scheduled_route = kinematic_path(scene, home, scheduled_q)
            scheduled_return = (
                kinematic_path(scene, scheduled_q, home)
                if scheduled_route["feasible"]
                else {"feasible": False, "duration_s": 0.0}
            )
            scheduled_seen = (
                scheduled_route["feasible"]
                and scheduled_return["feasible"]
                and scheduled_route["first_seen_s"] is not None
            )
            scene.set_episode(seed, phase)
            oracle, end = oracle_route(scene, catalog, oracle_start)
            oracle_start = end.copy()
            scene.set_episode(seed, phase)
            seen = False
            feasible = True
            duration = 0.0
            for idx in scan_pair:
                end = np.array(catalog[idx]["q"])
                route = kinematic_path(scene, scan_start, end)
                feasible &= route["feasible"]
                seen |= route["feasible"] and route["first_seen_s"] is not None
                duration += route["duration_s"]
                if not route["feasible"]:
                    break
                scan_start = end
            scene.set_episode(seed, phase)
            scene.set_camera_arm(scene.home["camera_arm"][:5])
            # A hand lift can reveal the same state; a real task must establish
            # whether unloading changes that state or violates continued contact.
            lift_collision = False
            for height in np.linspace(0, 0.045, 10):
                scene.set_manip_tcp(scene.expected_tcp + [0, 0, height])
                lift_collision |= bool(scene.external_contacts("manipulator"))
            retreat_seen = (
                scene.visibility(*fixed_view)["visible"]
                or scene.visibility(*scene.camera_view("manipulator/wrist_cam"))["visible"]
            )
            scene.set_episode(seed, phase)
            for mid in (scene.housing_id, scene.second_housing_id):
                if mid is not None:
                    scene.data.mocap_pos[mid, 2] = -1
            mujoco.mj_forward(scene.model, scene.data)
            clear_seen = (
                scene.visibility(*fixed_view)["visible"]
                or scene.visibility(*scene.camera_view("manipulator/wrist_cam"))["visible"]
            )
            records.append(
                {
                    "seed": seed,
                    "phase": phase,
                    "oracle_route": oracle,
                    "scan_feasible": bool(feasible),
                    "scan_seen": bool(seen and feasible),
                    "scan_duration_s": duration,
                    "scheduled_stow_seen": bool(scheduled_seen),
                    "scheduled_stow_feasible": bool(
                        scheduled_route["feasible"] and scheduled_return["feasible"]
                    ),
                    "scheduled_stow_roundtrip_s": scheduled_route["duration_s"]
                    + scheduled_return["duration_s"],
                    "hand_lift_seen": bool(retreat_seen),
                    "hand_lift_collision": bool(lift_collision),
                    "wait_until_housing_removed_seen": bool(clear_seen),
                }
            )

    def complete(key):
        return float(np.array([r[key] for r in records]).reshape(-1, 2).all(axis=1).mean())

    durations = [r["oracle_route"]["duration_s"] for r in records if r["oracle_route"]["feasible"]]
    return {
        "episodes": len(seeds),
        "queries": len(records),
        "oracle_route_query_coverage": float(
            np.mean([r["oracle_route"]["feasible"] for r in records])
        ),
        "oracle_route_episode_coverage": float(
            np.array([r["oracle_route"]["feasible"] for r in records])
            .reshape(-1, 2)
            .all(axis=1)
            .mean()
        ),
        "oracle_route_median_duration_s": float(np.median(durations)) if durations else None,
        "oracle_route_max_duration_s": max(durations) if durations else None,
        "scan_route_episode_coverage": complete("scan_seen"),
        "scheduled_stow_episode_coverage": complete("scheduled_stow_seen"),
        "scheduled_stow_feasible_episode_fraction": complete("scheduled_stow_feasible"),
        "scheduled_stow_median_roundtrip_s": float(
            np.median([r["scheduled_stow_roundtrip_s"] for r in records])
        ),
        "scan_route_feasible_episode_fraction": complete("scan_feasible"),
        "scan_median_full_cycle_s": float(np.median([r["scan_duration_s"] for r in records])),
        "hand_lift_episode_coverage": complete("hand_lift_seen"),
        "hand_lift_collision_queries": sum(r["hand_lift_collision"] for r in records),
        "wait_until_housing_removed_episode_coverage": complete("wait_until_housing_removed_seen"),
        "records": records,
    }


class Renderer:
    def __init__(self, scene, size):
        self.scene = scene
        self.renderer = mujoco.Renderer(scene.model, width=size[0], height=size[1])
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = GROUPS
        self.option.sitegroup[:] = 0

    def frame(self, camera, view=None, segmentation=False):
        scene = self.scene
        if view is not None:
            cid = scene.model.camera("screen_fixed").id
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, np.array(view["matrix"]).ravel())
            scene.model.cam_pos[cid] = view["position"]
            scene.model.cam_quat[cid] = quat
            mujoco.mj_forward(scene.model, scene.data)
            camera = "screen_fixed"
        if segmentation:
            self.renderer.enable_segmentation_rendering()
        else:
            self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(scene.data, camera=camera, scene_option=self.option)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def counterfactual(scene, seed, phase, catalog, fixed, output):
    """Move only the queried marker; report both direct pixels and shadow leakage."""
    scene.set_episode(seed, phase)
    home = scene.home["camera_arm"][:5].copy()
    route, q = oracle_route(scene, catalog, home)
    origin = scene.feature_center.copy()
    rows = []
    panels = []
    renderer = Renderer(scene, (96, 72))
    try:
        for name, view in (("wrist", None), ("searched_static", fixed), ("oracle_camera", None)):
            scene.set_camera_arm(q if name == "oracle_camera" else home)
            camera = "manipulator/wrist_cam" if name == "wrist" else "camera_arm/wrist_cam"
            pair = []
            pixels = []
            rays = []
            for dx in (-0.003, 0.003):
                scene.data.mocap_pos[scene.feature_id] = origin + [dx, 0, 0]
                scene.refresh_points()
                pair.append(renderer.frame(camera, view))
                seg = renderer.frame(camera, view, segmentation=True)
                pixels.append(
                    int(
                        (
                            (seg[:, :, 0] == scene.feature_geom)
                            & (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                        ).sum()
                    )
                )
                actual_view = (
                    (np.array(view["position"]), np.array(view["matrix"]))
                    if view
                    else scene.camera_view(camera)
                )
                rays.append(scene.visibility(*actual_view))
            diff = np.abs(pair[0].astype(int) - pair[1].astype(int)).max(axis=2)
            rows.append(
                {
                    "sensor": name,
                    "direct_marker_pixels": pixels,
                    "changed_rgb_pixels_gt_8": int((diff > 8).sum()),
                    "ray_visibility": rays,
                }
            )
            panels.append(pair)
    finally:
        renderer.close()
    canvas = Image.new("RGB", (3 * 384, 2 * 288 + 64), (20, 24, 30))
    draw = ImageDraw.Draw(canvas)
    for col, (row, pair) in enumerate(zip(rows, panels, strict=True)):
        for state, rgb in enumerate(pair):
            canvas.paste(
                Image.fromarray(rgb).resize((384, 288), Image.Resampling.NEAREST),
                (col * 384, state * 288),
            )
        draw.text(
            (col * 384 + 8, 584),
            f"{row['sensor']} | marker pixels {row['direct_marker_pixels']}",
            fill="white",
        )
        draw.text(
            (col * 384 + 8, 604),
            f"Changed RGB pixels: {row['changed_rgb_pixels_gt_8']}",
            fill="white",
        )
    canvas.save(output / f"{scene.candidate.name}_counterfactual.png")
    return {
        "seed": seed,
        "phase": phase,
        "oracle_index": route["index"],
        "marker_displacement_m": 0.006,
        "sensors": rows,
    }


def servo_video(scene, seed, catalog, fixed, output):
    """Real native camera servo; query phases are explicitly reset interventions."""
    scene.set_episode(seed, 0)
    scene.set_camera_arm(scene.home["camera_arm"][:5])
    wide = Renderer(scene, (1920, 1080))
    small = Renderer(scene, (96, 72))
    report = {
        "seed": seed,
        "phases": [],
        "stow_segments": [],
        "camera_collision_frames": 0,
        "nonfinite": False,
    }
    qstart = scene.home["camera_arm"][:5].copy()
    try:
        with (
            imageio.get_writer(
                output / f"{scene.candidate.name}_overview.mp4",
                fps=25,
                macro_block_size=2,
                codec="libx264",
                quality=8,
            ) as writer,
            imageio.get_writer(
                output / f"{scene.candidate.name}_sensors.mp4",
                fps=25,
                macro_block_size=2,
                codec="libx264",
                quality=8,
            ) as sensor_writer,
        ):
            for phase in (0, 1, 2):
                if phase:
                    # Stow under the OLD hand pose before the prescribed phase
                    # reset, which would otherwise move the hand through a camera.
                    home = scene.home["camera_arm"][:5].copy()
                    targets = qstart.copy()
                    stow_frames = int(np.ceil(np.max(np.abs(home - qstart)) / 0.035)) + 20
                    collisions_before = report["camera_collision_frames"]
                    for _frame in range(stow_frames):
                        targets += np.clip(home - targets, -0.035, 0.035)
                        scene.data.ctrl[scene.ctrlids["camera_arm"][:5]] = targets
                        mujoco.mj_step(scene.model, scene.data, nstep=20)
                        mujoco.mj_forward(scene.model, scene.data)
                        report["camera_collision_frames"] += bool(scene.camera_contacts())
                        report["nonfinite"] |= not bool(np.isfinite(scene.data.qpos).all())
                        canvas = Image.fromarray(wide.frame("overview"))
                        draw = ImageDraw.Draw(canvas)
                        draw.rectangle((0, 0, 1920, 56), fill=(15, 20, 28))
                        draw.text(
                            (18, 12),
                            f"VISIBILITY PROTOTYPE | {scene.candidate.name} | stowing before phase {phase}",
                            fill="white",
                            font_size=22,
                        )
                        writer.append_data(np.asarray(canvas))
                        tile_canvas = Image.new("RGB", (864, 258), (20, 24, 30))
                        td = ImageDraw.Draw(tile_canvas)
                        tiles = [
                            small.frame("manipulator/wrist_cam"),
                            small.frame("screen_fixed", fixed),
                            small.frame("camera_arm/wrist_cam"),
                        ]
                        for idx, (tile, label) in enumerate(
                            zip(
                                tiles,
                                ("Wrist", "Searched static", "Scripted camera arm"),
                                strict=True,
                            )
                        ):
                            tile_canvas.paste(
                                Image.fromarray(tile).resize((288, 216), Image.Resampling.NEAREST),
                                (idx * 288, 0),
                            )
                            td.text((idx * 288 + 8, 220), f"{label} | stowing", fill="white")
                        sensor_writer.append_data(np.asarray(tile_canvas))
                    qstart = scene.data.qpos[scene.qadr["camera_arm"][:5]].copy()
                    report["stow_segments"].append(
                        {
                            "before_phase": phase,
                            "frames": stow_frames,
                            "collision_frames": report["camera_collision_frames"]
                            - collisions_before,
                        }
                    )
                scene.set_episode(seed, phase)
                route, goal = oracle_route(scene, catalog, qstart)
                scene.set_camera_arm(qstart)
                scene.data.qvel[:] = 0
                targets = qstart.copy()
                frames = max(30, int(np.ceil(np.max(np.abs(goal - qstart)) / 0.035)) + 20)
                seen = 0
                for _frame in range(frames):
                    targets += np.clip(goal - targets, -0.035, 0.035)
                    scene.data.ctrl[scene.ctrlids["camera_arm"][:5]] = targets
                    mujoco.mj_step(scene.model, scene.data, nstep=20)
                    mujoco.mj_forward(scene.model, scene.data)
                    report["camera_collision_frames"] += bool(scene.camera_contacts())
                    report["nonfinite"] |= not bool(np.isfinite(scene.data.qpos).all())
                    visible = sensing(scene)
                    seen += visible
                    rgb = wide.frame("overview")
                    canvas = Image.fromarray(rgb)
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((0, 0, 1920, 56), fill=(15, 20, 28))
                    draw.text(
                        (18, 12),
                        f"VISIBILITY PROTOTYPE | {scene.candidate.name} | phase {phase} | seed {seed} | marker visible: {visible}",
                        fill="white",
                        font_size=22,
                    )
                    draw.text(
                        (18, 38),
                        "Prescribed query states; native camera servo; privileged camera destinations; no learned policy or task-success claim.",
                        fill="white",
                    )
                    writer.append_data(np.asarray(canvas))
                    tiles = [
                        small.frame("manipulator/wrist_cam"),
                        small.frame("screen_fixed", fixed),
                        small.frame("camera_arm/wrist_cam"),
                    ]
                    tile_canvas = Image.new("RGB", (864, 258), (20, 24, 30))
                    td = ImageDraw.Draw(tile_canvas)
                    for idx, (tile, label) in enumerate(
                        zip(tiles, ("Wrist", "Searched static", "Scripted camera arm"), strict=True)
                    ):
                        tile_canvas.paste(
                            Image.fromarray(tile).resize((288, 216), Image.Resampling.NEAREST),
                            (idx * 288, 0),
                        )
                        td.text((idx * 288 + 8, 220), f"{label} | phase {phase}", fill="white")
                    sensor_writer.append_data(np.asarray(tile_canvas))
                qstart = scene.data.qpos[scene.qadr["camera_arm"][:5]].copy()
                report["phases"].append(
                    {
                        "phase": phase,
                        "planned_route": route,
                        "frames": frames,
                        "visible_frames": seen,
                        "final_joint_error_rad": float(np.max(np.abs(qstart - goal))),
                    }
                )
                canvas.save(output / f"{scene.candidate.name}_phase{phase}.png")
    finally:
        wide.close()
        small.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/task_screening/pixel_screen/screening.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/task_screening/diagnostics"))
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--candidates", nargs="+", choices=[c.name for c in CANDIDATES])
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    source = json.loads(args.report.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.report),
        "interpretation": "Instantaneous visibility checks, without memory accumulation, and selected native camera-servo checks; not learned success. Oracle routes use true marker visibility. Paths sample commands at .035 rad and do not prove swept-volume safety. Hand retreat does not preserve contact; waiting removes the housing entirely. These are optimistic alternative-strategy controls. The phase schedule returns home before hand relocation; a blind scan may collide when trying the next phase's view prematurely.",
        "candidates": {},
    }
    for candidate in CANDIDATES:
        if candidate.name not in source["candidates"] or (
            args.candidates and candidate.name not in args.candidates
        ):
            continue
        scene = ScreenScene(candidate)
        stats = source["candidates"][candidate.name]
        catalog = source["active_catalog"]
        fixed = stats.get("fixed_catalog", source["fixed_catalog"])[stats["best_static_index"]]
        seeds = source["test_seeds"][: args.episodes]
        result = controls(
            scene,
            seeds,
            catalog,
            fixed,
            stats["best_validation_scan_pair"],
            stats["best_validation_scheduled_pair"],
        )
        if not args.no_render:
            result["counterfactual"] = counterfactual(
                scene, seeds[0], 1, catalog, fixed, args.output
            )
            result["servo_video"] = servo_video(scene, seeds[0], catalog, fixed, args.output)
        report["candidates"][candidate.name] = result
        print(
            candidate.name,
            {k: v for k, v in result.items() if not isinstance(v, (list, dict))},
            flush=True,
        )
        (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2))
        if not args.no_render:
            rendered = {
                "source": str(args.report),
                "candidates": {
                    name: {key: row[key] for key in ("counterfactual", "servo_video")}
                    for name, row in report["candidates"].items()
                },
            }
            (args.output / "rendered_checks.json").write_text(json.dumps(rendered, indent=2))


if __name__ == "__main__":
    main()
