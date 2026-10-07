"""Extract reported Warp module load/compile times without mislabeling total startup."""

import json
import re
from pathlib import Path

rows = []
for path in sorted(Path("logs/slurm").glob("plug-benchmark-*.out")):
    text = path.read_text()
    compiled = [float(x) / 1000 for x in re.findall(r"took ([\d.]+) ms\s+\(compiled\)", text)]
    cached = [float(x) / 1000 for x in re.findall(r"took ([\d.]+) ms\s+\(cached\)", text)]
    if compiled or cached:
        rows.append(
            dict(
                log=str(path),
                compiled_module_count=len(compiled),
                reported_compiled_load_seconds=sum(compiled),
                cached_module_count=len(cached),
                reported_cached_load_seconds=sum(cached),
            )
        )
Path("artifacts/cluster_pilot/compilation_timings.json").write_text(
    json.dumps(
        dict(
            note="Sum of Warp-reported module load times across inference and PPO processes in each job. "
            "Not total compilation wall time: Torch/Triton compilation and graph capture are not isolated "
            "and remain included in separately measured environment initialization and warmup. "
            "First wrist PPO replacement did not run inference and therefore had a cold cache.",
            rows=rows,
        ),
        indent=2,
    )
)
print("Saved", len(rows), "job compilation summaries")
