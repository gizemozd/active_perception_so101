#!/usr/bin/env bash
# Source from an sbatch script; invoke from the project root or set PROJECT_ROOT.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
# One cache per GPU job avoids concurrent compiler writes across array jobs.
export XDG_CACHE_HOME="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/active-perception-${SLURM_JOB_ID:-local}"
mkdir -p "$XDG_CACHE_HOME" logs/slurm artifacts
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing environment: run uv sync --locked once before submission.' >&2
  exit 1
fi
run_python() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    .venv/bin/python "$@"
  else
    srun --ntasks=1 .venv/bin/python "$@"
  fi
}
run_training() {
  local task="$1"
  local index="${SLURM_ARRAY_TASK_ID:-0}"
  local conditions=(wrist static wrist_static initial scheduled active)
  if (( index < 0 || index >= 18 )); then echo 'Expected array index 0..17' >&2; exit 2; fi
  local condition="${conditions[$((index / 3))]}"
  local seed="$((index % 3))"
  # Same environment count, transitions and PPO settings across GPU types/conditions.
  local nenv="${NUM_ENVS:-256}"
  local budget="${TOTAL_STEPS:-18432000}"
  if (( nenv % 8 != 0 || budget % (nenv * 24) != 0 )); then
    echo 'NUM_ENVS must divide 8 minibatches; TOTAL_STEPS must divide NUM_ENVS*24.' >&2; exit 2
  fi
  local view=0
  local default_occlusion=random
  if [[ "$task" == plug ]]; then default_occlusion=clean; fi
  if [[ "$condition" == static || "$condition" == wrist_static ]]; then view="${FIXED_VIEW:-0}"; fi
  local args=(-m active_perception_arms.train --task "$task" --condition "$condition"
    --seed "$seed" --num-envs "$nenv" --iterations "$((budget / (nenv * 24)))"
    --occlusion "${OCCLUSION:-$default_occlusion}" --memory "${MEMORY:-gru}" --fixed-view "$view"
    --log-root "${LOG_ROOT:-logs}" --job-type "${JOB_TYPE:-pilot}")
  if [[ -n "${RESULT_ROOT:-}" ]]; then args+=(--result "${RESULT_ROOT}/${task}-${condition}-s${seed}.json"); fi
  if [[ "${PERTURB_PUSH:-0}" == 1 ]]; then args+=(--perturb-push); fi
  if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
  if [[ "${DRY_RUN:-0}" == 1 ]]; then args+=(--dry-run); fi
  printf 'task=%s condition=%s seed=%s envs=%s budget=%s fixed_view=%s\n' "$task" "$condition" "$seed" "$nenv" "$budget" "$view"
  run_python "${args[@]}"
}
