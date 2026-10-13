"""Matched fresh-test controls for the push environment repair."""

import json
import subprocess
import sys
from pathlib import Path

root = Path("artifacts/competence_repair")
source = Path(
    json.loads(Path("artifacts/push_rgb_arc120/runs/push_wrist_static_s2/state.json").read_text())[
        "checkpoint"
    ]
)
for name, checkpoint in [
    ("push_original", source),
    ("push_wall_ablation", root / "push_wall_ablation/model.pt"),
]:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "active_perception_arms.evaluate",
            str(checkpoint),
            "--episodes",
            "1024",
            "--num-envs",
            "128",
            "--seed",
            "90000",
            "--split",
            "test",
            "--occlusion",
            "clean",
            "--audit-termination",
            "--output",
            str(out / "final-test.json"),
        ],
        check=True,
    )
