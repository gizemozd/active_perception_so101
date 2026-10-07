#!/usr/bin/env bash
#SBATCH --job-name=plug-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%1
#SBATCH --output=logs/slurm/%x-%A_%a.out
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
conditions=(wrist wrist_static initial active)
read -r -a counts <<< "${BENCH_COUNTS:-64 128 256 512}"
index="${SLURM_ARRAY_TASK_ID:-0}"
if (( index < 0 || index >= 4 * ${#counts[@]} )); then
  echo 'Array index exceeds condition/count grid' >&2; exit 2
fi
condition="${conditions[$((index / ${#counts[@]}))]}"
count="${counts[$((index % ${#counts[@]}))]}"
view=0
if [[ "$condition" == wrist_static ]]; then view=7; fi
output="artifacts/cluster_pilot/bench-${SLURM_JOB_ID:-local}-${condition}-${count}"
if [[ "${SKIP_INFERENCE:-0}" != 1 ]]; then
  run_python -m active_perception_arms.benchmark --task plug --condition "$condition" \
    --fixed-view "$view" --num-envs "$count" --actor --wandb --output "${output}-inference.json"
fi
run_python -m active_perception_arms.train --task plug --condition "$condition" \
  --fixed-view "$view" --num-envs "$count" --memory gru --occlusion clean --seed 0 \
  --iterations "${BENCH_ITERATIONS:-12}" --warmup-iterations 3 --job-type benchmark \
  --log-root logs/benchmarks --result "${output}-ppo.json"
