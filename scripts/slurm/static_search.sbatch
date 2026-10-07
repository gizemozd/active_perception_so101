#!/usr/bin/env bash
#SBATCH --job-name=vision-static-search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@180
#SBATCH --array=0-77%6
#SBATCH --output=logs/slurm/%x-%A_%a.out
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
: "${TASK:?Set TASK to plug, transfer, or push}"
condition="${CONDITION:-wrist_static}"
default_occlusion=random
if [[ "$TASK" == plug ]]; then default_occlusion=clean; fi
if [[ "$condition" != static && "$condition" != wrist_static ]]; then exit 2; fi
index="${SLURM_ARRAY_TASK_ID:-0}"
nenv="${NUM_ENVS:-256}"
budget="${TOTAL_STEPS:-18432000}"
if (( index < 0 || index > 77 || nenv % 8 != 0 || budget % (nenv * 24) != 0 )); then exit 2; fi
args=(-m active_perception_arms.train --task "$TASK" --condition "$condition"
  --seed "$((index % 3))" --fixed-view "$((index / 3))" --num-envs "$nenv"
  --iterations "$((budget / (nenv * 24)))" --occlusion "${OCCLUSION:-$default_occlusion}"
  --memory "${MEMORY:-gru}" --log-root "${LOG_ROOT:-logs/static-search}" --job-type "${JOB_TYPE:-study}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then args+=(--dry-run); fi
run_python "${args[@]}"
