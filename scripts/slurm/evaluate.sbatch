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
run_python -m active_perception_arms.evaluate "$CHECKPOINT" --episodes "${EPISODES:-512}" \
 --num-envs "${NUM_ENVS:-128}" --split "${SPLIT:-test}" \
 --output "${OUTPUT:-artifacts/evaluation-${SLURM_JOB_ID:-local}.json}"
