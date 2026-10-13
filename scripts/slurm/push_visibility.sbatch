#!/usr/bin/env bash
#SBATCH --job-name=push-visibility
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
root="$1"
cd "$root"
exec "$root/.venv/bin/python" scripts/audit_push_visibility.py --output artifacts/push_rgb/visibility-audit --seeds 64
