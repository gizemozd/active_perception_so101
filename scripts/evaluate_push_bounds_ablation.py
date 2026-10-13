"""Separate frozen-policy command-range changes from unsuccessful PPO fine-tuning."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from active_perception_arms.manipulation import bounds

root = Path("artifacts/competence_repair")
state = json.loads(
    Path("artifacts/push_rgb_arc120/runs/push_wrist_static_s2/state.json").read_text()
)
source = Path(state["checkpoint"])
for remap in [False, True]:
    folder = root / ("push_frozen_wide_remap" if remap else "push_frozen_wide_direct")
    folder.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(source.with_name("experiment.json").read_text())
    cfg.update(push_workspace_revision="arc_clear_v2", push_control_revision="wide_v2")
    (folder / "experiment.json").write_text(json.dumps(cfg, indent=2))
    shutil.copyfile(source.with_name("runner.json"), folder / "runner.json")
    if remap:
        weights = torch.load(source, map_location="cpu", weights_only=True)
        actor = weights["actor_state_dict"]
        layer = max(
            int(k.split(".")[1]) for k in actor if k.startswith("mlp.") and k.endswith(".weight")
        )
        old_low, old_high = [torch.tensor(x) for x in bounds("push")]
        new_low, new_high = [torch.tensor(x) for x in bounds("push", "wide_v2")]
        scale = (old_high - old_low) / (new_high - new_low)
        offset = (old_high + old_low - new_high - new_low) / (new_high - new_low)
        actor[f"mlp.{layer}.weight"][:3].mul_(scale[:, None])
        actor[f"mlp.{layer}.bias"][:3].mul_(scale).add_(offset)
        torch.save(weights, folder / "model.pt")
    elif not (folder / "model.pt").exists():
        (folder / "model.pt").symlink_to(source)
    (folder / "intervention.json").write_text(
        json.dumps(
            dict(
                source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                remap=remap,
                description="Frozen source policy; expanded command bounds and partition removal; optional affine actor head remap, no training",
            ),
            indent=2,
        )
    )
    subprocess.run(
        [
            sys.executable,
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
            str(folder / "validation.json"),
        ],
        check=True,
    )
