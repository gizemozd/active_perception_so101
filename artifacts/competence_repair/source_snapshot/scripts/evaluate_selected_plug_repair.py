"""Select the occlusion repair using validation only, then test untouched resets."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

folder = Path("artifacts/competence_repair/plug_dense_ym_dynamic")
candidates = []
for step in [2000, 4000, 6000, 8000]:
    clean = json.loads((folder / f"evaluation-{step}.json").read_text())
    dynamic = json.loads((folder / f"evaluation-dynamic-{step}.json").read_text())
    assert clean["split"] == dynamic["split"] == "validation"
    assert clean["seed"] == dynamic["seed"] == 31000
    if clean["success_rate"] >= 0.8:
        candidates.append((dynamic["success_rate"], clean["success_rate"], -step))
assert candidates
best = max(candidates)
step = -best[2]
checkpoint = folder / f"model_{step}.pt"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
selection = dict(
    checkpoint=str(checkpoint),
    sha256=digest,
    selected_step=step,
    selection_rule="Highest dynamic validation success among checkpoints with clean validation >=80%; ties prefer clean success then earlier step.",
    validation_candidates=candidates,
    test_seed=91000,
    rationale_for_new_seed="Seed90000 exposed the original occlusion weakness;91000 reserved for the updated model.",
)
(folder / "selected_model.json").write_text(json.dumps(selection, indent=2) + "\n")
print(json.dumps(selection), flush=True)
for occlusion in ["clean", "dynamic"]:
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
            "91000",
            "--split",
            "test",
            "--occlusion",
            occlusion,
            "--audit-termination",
            "--capture-representatives",
            str(folder / f"final_{occlusion}_capture"),
            "--output",
            str(folder / f"final-{occlusion}-test.json"),
        ],
        check=True,
    )
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    subprocess.run(
        [
            sys.executable,
            "-m",
            "active_perception_arms.record_evaluation",
            str(folder / f"final-{occlusion}-test.json"),
            "--output",
            str(folder / f"final_{occlusion}_videos"),
        ],
        check=True,
    )
