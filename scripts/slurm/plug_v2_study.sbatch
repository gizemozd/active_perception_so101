#!/usr/bin/env bash
# hidden_prongs_v2 camera comparison: seated success, jam failure, bounded staged
# reward, optical gap screen, absolute look-at camera and auxiliary perception loss.
# Array index -> condition (wrist, wrist_static, initial, active) x seed (0, 1, 2).
#
#   sbatch --account=ACCOUNT --partition=PARTITION scripts/slurm/plug_v2_study.sbatch
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=11 bash scripts/slurm/plug_v2_study.sbatch
#
# Overrides: NUM_ENVS, TOTAL_STEPS, SCREEN (gap_v1|none), AUX_COEF, CAMERA_REWARD,
# RESET_SCALE, ASSIST, LOG_ROOT, RESUME, ACTOR_OBSERVATION (rgb|privileged),
# ACTION_DISTRIBUTION (squashed|gaussian). Squashed bounds commands and std; the
# unbounded Gaussian saturated 84-100% of camera commands in earlier studies.
#SBATCH --job-name=plug-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@180
#SBATCH --array=0-11%8
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"

index="${SLURM_ARRAY_TASK_ID:-0}"
conditions=(wrist wrist_static initial active)
if (( index < 0 || index >= 12 )); then echo 'Expected array index 0..11' >&2; exit 2; fi
condition="${conditions[$((index / 3))]}"
seed="$((index % 3))"
nenv="${NUM_ENVS:-1024}"
# 50,012,160 transitions = 2,035 PPO updates x 1,024 worlds x 24 steps.
budget="${TOTAL_STEPS:-50012160}"
if (( nenv % 8 != 0 || budget % (nenv * 24) != 0 )); then
  echo 'NUM_ENVS must divide 8 minibatches; TOTAL_STEPS must divide NUM_ENVS*24.' >&2; exit 2
fi
args=(-m active_perception_arms.train --task plug --condition "$condition" --seed "$seed"
  --task-revision hidden_prongs_v2 --control-revision pose_v1 --translation-control delta
  --camera-action lookat_v2 --plug-screen "${SCREEN:-gap_v1}"
  --plug-reset-scale "${RESET_SCALE:-1.0}" --plug-assist-fraction "${ASSIST:-0.3}"
  --occlusion clean --memory gru
  --actor-observation "${ACTOR_OBSERVATION:-rgb}"
  --action-distribution "${ACTION_DISTRIBUTION:-squashed}"
  --initial-std "${INITIAL_STD:-0.4}" --entropy-coef "${ENTROPY_COEF:-0.003}"
  --num-envs "$nenv" --iterations "$((budget / (nenv * 24)))" --save-interval 200
  --log-root "${LOG_ROOT:-logs/plug_v2}" --job-type study
  --run-label "v2_${SCREEN:-gap_v1}")
if [[ "${ACTOR_OBSERVATION:-rgb}" == rgb ]]; then
  # Privileged labels train an auxiliary head only; they are never actor inputs.
  args+=(--aux-loss-coef "${AUX_COEF:-0.5}")
fi
if [[ "$condition" == initial || "$condition" == active ]]; then
  # Training-only shaping for the controllable camera; zero disables it.
  args+=(--camera-visibility-reward "${CAMERA_REWARD:-0.02}")
fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then args+=(--dry-run); fi
printf 'plug-v2 condition=%s seed=%s envs=%s budget=%s screen=%s\n' \
  "$condition" "$seed" "$nenv" "$budget" "${SCREEN:-gap_v1}"
run_python "${args[@]}"
