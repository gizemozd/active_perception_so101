#!/usr/bin/env bash
#SBATCH --job-name=push-vis-warp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
root="$1"
export PYTHONPATH="$root/src:$root/scripts"
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="/tmp/push-vis-warp-${SLURM_JOB_ID}"
cd "$root"
exec "$root/.venv/bin/python" scripts/audit_push_visibility_warp.py --output artifacts/push_rgb/warp-audit --num-envs 128
