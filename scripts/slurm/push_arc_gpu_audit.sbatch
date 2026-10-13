#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="/tmp/push-arc-audit-${SLURM_JOB_ID}"
export PYTHONPATH="$1/src"
exec "$2" "$1/scripts/audit_push_arc_gpu.py" --output "$3" --episodes 64
