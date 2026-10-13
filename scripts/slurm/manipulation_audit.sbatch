#!/usr/bin/env bash
#SBATCH --job-name=transfer-push-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
root="$1"
snapshot="$2"
output="$3"
export PYTHONPATH="$snapshot/src"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="/tmp/manipulation-audit-${SLURM_JOB_ID}"
cd "$snapshot"
sha256sum --quiet -c sha256sum.txt
exec "$root/.venv/bin/python" scripts/audit_manipulation_study.py --output "$output"
