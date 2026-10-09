#!/usr/bin/env bash
#SBATCH --job-name=plug-success-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT}/scripts/slurm/common.sh"
: "${DATA_ROOT:?Set DATA_ROOT to the original results checkout}"
for condition in initial active; do
  checkpoint=$("$DATA_ROOT/.venv/bin/python" - "$DATA_ROOT" "$condition" <<'PY'
import json, sys
from pathlib import Path
root, condition = Path(sys.argv[1]), sys.argv[2]
manifest = json.loads((root / "artifacts/cluster_continuation/manifest.json").read_text())
record = next(row for row in manifest if row["condition"] == condition)
print(root / record["expected_final_checkpoint"])
PY
  )
  reference="$DATA_ROOT/artifacts/cluster_continuation/evaluation-${condition}-s0.json"
  for repeat in 1 2; do
    output="$DATA_ROOT/artifacts/plug_analysis/termination-${condition}-repeat${repeat}.json"
    run_python -m active_perception_arms.evaluate "$checkpoint" --episodes 512 \
      --num-envs 128 --seed 10000 --split validation --audit-termination \
      --reference-report "$reference" --output "$output"
    reference="$output"
  done
done
