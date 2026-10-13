#!/usr/bin/env bash
# Bounded simulator/reward validation. No competence-training run.
#SBATCH --job-name=transfer-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
root="$1"
snapshot="$2"
output="$3"
export PYTHONPATH="$snapshot/src"
export MUJOCO_GL=egl
export MPLCONFIGDIR="/tmp/transfer-audit-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
cd "$snapshot"
sha256sum --quiet -c sha256sum.txt
exec "$root/.venv/bin/python" "$snapshot/audit_transfer_warp.py" \
    --output "$output" --num-envs 64 --seeds 10000 10001 --render-videos
