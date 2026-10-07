"""Produce an unsubmitted larger-study budget from measured PPO benchmark records."""

import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("benchmarks", nargs="+", type=Path)
p.add_argument("--num-envs", type=int, required=True)
p.add_argument("--concurrency", type=int, default=4)
p.add_argument("--output", type=Path, default=Path("artifacts/cluster_pilot/study_budget.json"))
a = p.parse_args()
rates = {}
for path in a.benchmarks:
    r = json.loads(path.read_text())
    if r.get("training_started") and r["experiment"]["num_envs"] == a.num_envs:
        rates[r["experiment"]["condition"]] = r["steady_transitions_per_second"]
assert all(c in rates for c in ("wrist", "wrist_static", "initial", "active"))
budget = 18432000
cells = [
    ("26-view wrist+static validation search", "wrist_static", 78, False),
    ("wrist, three seeds", "wrist", 3, False),
    ("initial-only, three seeds", "initial", 3, False),
    ("active, three seeds", "active", 3, False),
    ("scheduled motion, three seeds", "active", 3, True),
    ("feedforward initial+active, three seeds each", "active", 6, True),
    ("trained initial snapshot, three seeds", "initial", 3, True),
]
rows = []
for label, condition, runs, optional in cells:
    seconds = budget / rates[condition]
    rows.append(
        {
            "comparison": label,
            "runs": runs,
            "transitions": runs * budget,
            "estimated_gpu_hours": runs * seconds / 3600,
            "estimated_elapsed_hours": math.ceil(runs / a.concurrency) * seconds / 3600,
            "optional": optional,
            "timing_proxy_condition": condition,
        }
    )
result = {
    "submitted": False,
    "num_envs": a.num_envs,
    "concurrency": a.concurrency,
    "transitions_per_policy": budget,
    "rows": rows,
    "required_transitions": sum(r["transitions"] for r in rows if not r["optional"]),
    "required_gpu_hours": sum(r["estimated_gpu_hours"] for r in rows if not r["optional"]),
    "assumptions": "Short benchmark extrapolation, excludes queues/startup/evaluation. "
    "Ablation rates are proxies. Search selects views by mean validation success over "
    "three seeds. Frozen-camera/live-image and frozen-camera/stale-image interventions "
    "use existing checkpoints and add evaluation cost, not training transitions.",
}
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
