#!/usr/bin/env bash
# Compare all four model_750 checkpoints under current physical-state scoring.
# No optimizer updates; preserve the original historical-criterion evaluations.
#SBATCH --job-name=plug-hparam-physical
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT}/scripts/slurm/common.sh"
: "${DATA_ROOT:?Set DATA_ROOT to the results checkout}"
for case in baseline fixed_lr entropy both; do
  checkpoint=$("$DATA_ROOT/.venv/bin/python" - "$DATA_ROOT" "$case" <<'PY'
import json, sys
from pathlib import Path
import torch
root, case = Path(sys.argv[1]), sys.argv[2]
plan = json.loads((root / "artifacts/hparam_search/next_stage_plan.json").read_text())
if case == "baseline":
    checkpoint = root / plan["baseline_checkpoint"]
else:
    row = next(row for row in plan["cases"] if row["case"] == case)
    checkpoint = Path(row["expected_final_checkpoint"])
assert torch.load(checkpoint, map_location="cpu", weights_only=True)["iter"] == 750
print(checkpoint)
PY
  )
  directory="$DATA_ROOT/artifacts/hparam_search/continuation/$case"
  mkdir -p "$directory"
  for repeat in 1 2; do
    output="$directory/corrected-evaluation-repeat${repeat}.json"
    [[ ! -e "$output" ]] || { echo "Existing result: $output; inspect rather than repeat"; exit 2; }
    reference=()
    if [[ "$repeat" == 2 ]]; then
      reference=(--reference-report "$directory/corrected-evaluation-repeat1.json")
    fi
    run_python -m active_perception_arms.evaluate "$checkpoint" --episodes 512 \
      --num-envs 128 --seed 10000 --split validation \
      --success-state-sample current_qpos --audit-termination \
      --output "$output" "${reference[@]}" --wandb
  done
done
