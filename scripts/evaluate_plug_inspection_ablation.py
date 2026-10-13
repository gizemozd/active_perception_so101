"""Validate when to freeze the RGB estimate, without changing learned weights."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

root = Path("artifacts/competence_repair")
source = (root / "plug_dense_ym/model_6000.pt").resolve()
for end in [0.72, 1.2, 1.6]:
    out = root / f"plug_freeze_{end}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.with_name("experiment.json"), out / "experiment.json")
    options = json.loads(source.with_name("runner.json").read_text())
    options["actor"]["inspection_end"] = end
    (out / "runner.json").write_text(json.dumps(options, indent=2))
    if not (out / "model.pt").exists():
        (out / "model.pt").symlink_to(source)
    for occlusion in ["clean", "dynamic"]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "active_perception_arms.evaluate",
                str(out / "model.pt"),
                "--episodes",
                "256",
                "--num-envs",
                "128",
                "--seed",
                "31000",
                "--split",
                "validation",
                "--occlusion",
                occlusion,
                "--audit-termination",
                "--output",
                str(out / f"validation-{occlusion}.json"),
            ],
            check=True,
        )
