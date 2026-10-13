"""Matched partition-removal tests across the completed push population."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--condition", choices=["wrist", "wrist_static", "active"], required=True)
a = p.parse_args()
for seed in range(3):
    run = f"push_{a.condition}_s{seed}"
    state = json.loads((Path("artifacts/push_rgb_arc120/runs") / run / "state.json").read_text())
    source = Path(state["checkpoint"])
    for intervention in ["original", "clear"]:
        out = Path("artifacts/competence_repair/push_population") / run / intervention
        out.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(source.with_name("experiment.json").read_text())
        if intervention == "clear":
            cfg["push_workspace_revision"] = "arc_clear_v2"
        (out / "experiment.json").write_text(json.dumps(cfg, indent=2))
        shutil.copyfile(source.with_name("runner.json"), out / "runner.json")
        if not (out / "model.pt").exists():
            (out / "model.pt").symlink_to(source)
        (out / "provenance.json").write_text(
            json.dumps(
                dict(
                    source=str(source),
                    sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    intervention=intervention,
                ),
                indent=2,
            )
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "active_perception_arms.evaluate",
                str(out / "model.pt"),
                "--episodes",
                "512",
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
