#!/usr/bin/env bash
#SBATCH --job-name=vision-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
run_python -m pytest -q tests/test_mjlab.py tests/test_policy.py
run_python -m active_perception_arms.benchmark --task plug --num-envs 32 --steps 25 --actor \
 --output "artifacts/gpu-smoke-${SLURM_JOB_ID:-local}.json"
