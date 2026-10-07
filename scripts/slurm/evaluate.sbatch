#!/usr/bin/env bash
#SBATCH --job-name=vision-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
: "${CHECKPOINT:?Set CHECKPOINT to a model .pt file}"
output="${OUTPUT:-artifacts/cluster_pilot/evaluation-${SLURM_JOB_ID:-local}.json}"
run_python -m active_perception_arms.evaluate "$CHECKPOINT" --episodes "${EPISODES:-512}" \
 --num-envs "${NUM_ENVS:-128}" --split "${SPLIT:-validation}" --wandb \
 --output "$output"
if [[ "${RECORD_VIDEO:-0}" == 1 ]]; then
 run_python -m active_perception_arms.record_evaluation "$output" --output "${output%.json}-videos" --wandb
fi
