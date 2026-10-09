#!/usr/bin/env bash
#SBATCH --job-name=plug-corrected-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT}/scripts/slurm/common.sh"
: "${DATA_ROOT:?Set DATA_ROOT to the results checkout}"
run_python -m pytest tests/test_dynamic_occlusion.py tests/test_mjlab.py -m gpu -q
"$DATA_ROOT/.venv/bin/python" - "$DATA_ROOT" "$SLURM_JOB_ID" <<'PY'
import json, sys
from pathlib import Path
root, job = Path(sys.argv[1]), sys.argv[2]
(root / "artifacts/dynamic_occlusion/gpu_validation.json").write_text(json.dumps({
    "job_id": job, "source_revision": "5badc6e", "pytest_passed": True,
    "modules": ["test_dynamic_occlusion.py", "test_mjlab.py"],
    "checks": "CUDA RGB, midpoint moving panel parity, partial resets, original pipeline regressions",
}, indent=2) + "\n")
PY
for condition in wrist wrist_static initial active; do
  checkpoint=$("$DATA_ROOT/.venv/bin/python" - "$DATA_ROOT" "$condition" <<'PY'
import json, sys
from pathlib import Path
root, condition = Path(sys.argv[1]), sys.argv[2]
record = next(row for row in json.loads((root / "artifacts/cluster_continuation/manifest.json").read_text()) if row["condition"] == condition)
print(root / record["expected_final_checkpoint"])
PY
  )
  output="$DATA_ROOT/artifacts/plug_analysis/corrected-${condition}-s0.json"
  run_python -m active_perception_arms.evaluate "$checkpoint" --episodes 512 \
    --num-envs 128 --seed 10000 --split validation --success-state-sample current_qpos \
    --audit-termination --reference-report "$DATA_ROOT/artifacts/cluster_continuation/evaluation-${condition}-s0.json" \
    --capture-representatives "${output%.json}-traces" --output "$output" --wandb
  run_python -m active_perception_arms.record_evaluation "$output" \
    --output "${output%.json}-videos" --wandb
done
