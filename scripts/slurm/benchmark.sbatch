#!/usr/bin/env bash
#SBATCH --job-name=vision-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/slurm/%x-%A_%a.out
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
tasks=(plug transfer push)
counts=(64 128 256 512)
index="${SLURM_ARRAY_TASK_ID:-0}"
task="${tasks[$((index / 4))]}"
count="${counts[$((index % 4))]}"
run_python -m active_perception_arms.benchmark --task "$task" --num-envs "$count" --actor \
  --output "artifacts/benchmark-${SLURM_JOB_ID:-local}-${task}-${count}.json"
