"""Matched validation of unchanged push weights with a longer episode budget."""

import json
import os
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "artifacts/competence_repair/push_horizon"
out.mkdir(exist_ok=True, parents=True)
s = json.loads(
    (root / "artifacts/push_rgb_arc120/runs/push_wrist_static_s2/state.json").read_text()
)
checkpoint = Path(s["checkpoint"])
for horizon in [12, 20, 30]:
    folder = out / str(horizon)
    folder.mkdir(exist_ok=True)
    cfg = json.loads(checkpoint.with_name("experiment.json").read_text())
    cfg["episode_seconds"] = horizon
    (folder / "experiment.json").write_text(json.dumps(cfg, indent=2))
    shutil.copyfile(checkpoint.with_name("runner.json"), folder / "runner.json")
    if not (folder / "model.pt").exists():
        (folder / "model.pt").symlink_to(checkpoint)
    (folder / "intervention.json").write_text(
        json.dumps(
            dict(
                source_checkpoint=str(checkpoint),
                source_sha256=s["checkpoint_sha256"],
                intervention="episode_seconds",
                value=horizon,
            ),
            indent=2,
        )
    )
    subprocess.run(
        [
            os.sys.executable,
            "-m",
            "active_perception_arms.evaluate",
            str(folder / "model.pt"),
            "--episodes",
            "512",
            "--num-envs",
            "128",
            "--seed",
            "31000",
            "--split",
            "validation",
            "--occlusion",
            "clean",
            "--audit-termination",
            "--output",
            str(folder / "evaluation.json"),
        ],
        check=True,
    )
