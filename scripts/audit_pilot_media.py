"""Decode recorded pilot videos and summarize selected actual trajectories."""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

root = Path("artifacts/cluster_pilot")
checks, diagnostics, gallery = [], [], []
for path in sorted(root.glob("capture-evaluation-*-s0-videos/representatives.json")):
    media = json.loads(path.read_text())
    report = json.loads(
        path.parent.with_name(path.parent.name.removesuffix("-videos") + ".json").read_text()
    )
    condition = report["experiment"]["condition"]
    seen = set()
    for record in media["records"]:
        assert record["expected_success"] == report["outcomes"][record["episode"]]
        decoded = []
        for kind in ("outside", "policy"):
            video = Path(record[kind + "_video"])
            reader = imageio.get_reader(video)
            n = reader.count_frames()
            meta = reader.get_meta_data()
            for index in (0, n // 2, n - 1):
                frame = reader.get_data(index)
            assert n == len(record["steps"]) and abs(meta["fps"] - 25) < 1e-6
            if kind == "outside":
                assert tuple(meta["size"]) == (1920, 1080)
            decoded.append(Image.fromarray(reader.get_data(max(0, n - 2))))
            reader.close()
            checks.append(
                dict(
                    path=str(video),
                    frames=n,
                    fps=meta["fps"],
                    size=meta["size"],
                    decoded_first_middle_last=True,
                )
            )
        errors = [x["position_error_m"] for x in record["steps"]]
        actions = np.array([x["action"] for x in record["steps"]])
        diagnostics.append(
            dict(
                condition=condition,
                variant=record["variant"],
                episode=record["episode"],
                success=record["expected_success"],
                minimum_error_mm=1000 * min(errors),
                last_preterminal_error_mm=1000 * errors[-1],
                fraction_manip_commands_clipped=float((np.abs(actions[:, :3]) > 1).mean()),
                camera_target_max_change_after_inspection=record.get(
                    "camera_target_max_change_after_inspection"
                ),
            )
        )
        if record["expected_success"] not in seen:
            seen.add(record["expected_success"])
            gallery.append(
                (
                    f"{condition} {record['variant']} {'success' if record['expected_success'] else 'failure'} (recorded evaluation)",
                    decoded,
                )
            )
canvas = Image.new("RGB", (1060, 325 * len(gallery)), "white")
draw = ImageDraw.Draw(canvas)
for i, (label, images) in enumerate(gallery):
    draw.text((8, i * 325 + 5), label + " | outside / actual policy input", fill="black")
    for j, im in enumerate(images):
        im.thumbnail((520, 295))
        canvas.paste(im, (j * 530, i * 325 + 25))
canvas.save(root / "recorded_representative_frames.png")
(root / "media_validation.json").write_text(json.dumps(checks, indent=2))
(root / "trajectory_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
print("Verified", len(checks), "videos;", len(diagnostics), "actual trajectories")
