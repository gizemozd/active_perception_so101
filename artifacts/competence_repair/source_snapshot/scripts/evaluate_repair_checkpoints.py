"""Audited validation of already-written intermediate repair checkpoints."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("folders", nargs="+", type=Path)
p.add_argument("--iteration", type=int, default=250)
a = p.parse_args()
for folder in a.folders:
    candidates = list((folder / "logs").glob(f"*/model_{a.iteration}.pt"))
    assert len(candidates) == 1, (folder, candidates)
    checkpoint = candidates[0]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = folder / f"intermediate-{a.iteration}.json"
    if output.exists():
        continue
    subprocess.run(
        [
            os.sys.executable,
            "-m",
            "active_perception_arms.evaluate",
            str(checkpoint),
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
            str(output),
        ],
        check=True,
    )
    assert digest == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (folder / f"intermediate-{a.iteration}-hash.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "sha256": digest}, indent=2)
    )
