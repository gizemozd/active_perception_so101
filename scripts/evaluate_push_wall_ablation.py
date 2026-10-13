"""Evaluate identical learned weights with only the obsolete partition removed."""

import json
import os
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "artifacts/competence_repair/push_wall_ablation"
out.mkdir(exist_ok=True, parents=True)
s = json.loads(
    (root / "artifacts/push_rgb_arc120/runs/push_wrist_static_s2/state.json").read_text()
)
checkpoint = Path(s["checkpoint"])
cfg = json.loads(checkpoint.with_name("experiment.json").read_text())
cfg["push_workspace_revision"] = "arc_clear_v2"
(out / "experiment.json").write_text(json.dumps(cfg, indent=2))
shutil.copyfile(checkpoint.with_name("runner.json"), out / "runner.json")
if not (out / "model.pt").exists():
    (out / "model.pt").symlink_to(checkpoint)
(out / "intervention.json").write_text(
    json.dumps(
        dict(
            source_checkpoint=str(checkpoint),
            source_sha256=s["checkpoint_sha256"],
            intervention="partition removal only",
            baseline="../push_horizon/12/evaluation.json",
        ),
        indent=2,
    )
)
subprocess.run(
    [
        os.sys.executable,
        "-m",
        "active_perception_arms.evaluate",
        str(out / "model.pt"),
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
        str(out / "evaluation.json"),
    ],
    check=True,
)
