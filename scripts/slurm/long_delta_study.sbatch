#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/long-delta-${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME"
plan="$1"
action="$2"
source_dir="$(dirname "$plan")/source"
if [[ "$action" == gate ]]; then
  exec .venv/bin/python "$source_dir/run_long_delta_study.py" gate --plan "$plan"
fi
if [[ "$action" == review ]]; then
  exec .venv/bin/python "$source_dir/run_long_delta_study.py" review --plan "$plan"
fi
if [[ "$action" == extend ]]; then
  exec .venv/bin/python "$source_dir/run_long_delta_study.py" run --plan "$plan" --index "$SLURM_ARRAY_TASK_ID" --extend
fi
exec .venv/bin/python "$source_dir/run_long_delta_study.py" run --plan "$plan" --index "$SLURM_ARRAY_TASK_ID"
