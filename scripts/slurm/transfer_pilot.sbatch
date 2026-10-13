#!/usr/bin/env bash
#SBATCH --job-name=transfer-v2-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/transfer-pilot-${SLURM_JOB_ID}"
exec "$1" "$2" --plan "$3"
