#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="/tmp/common-control-${SLURM_JOB_ID}"
exec .venv/bin/python "$2/run_common_control_study.py" run --plan "$1" --index "$SLURM_ARRAY_TASK_ID"
