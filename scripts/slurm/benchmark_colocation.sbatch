#!/usr/bin/env bash
#SBATCH --job-name=plug-colocation
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
# Three disposable active/GRU measurements: one process, then two independent
# processes on different GPUs of the same node. Never a distributed policy.
count="${NUM_ENVS:-512}"
export WANDB_RUN_GROUP=plug-colocation-20261007
run_cell() {
  local cell="$1"
  srun --exclusive --ntasks=1 --cpus-per-task=8 --mem=48G --gres=gpu:1 \
    env XDG_CACHE_HOME="$XDG_CACHE_HOME/$cell" \
    WANDB_TAGS="restored-plug,benchmark,colocation,$cell,active,gru" \
    .venv/bin/python -m active_perception_arms.train --task plug --condition active \
    --num-envs "$count" --memory gru --occlusion clean --seed 0 \
    --iterations 12 --warmup-iterations 3 --job-type benchmark \
    --log-root "logs/benchmarks/colocation/$SLURM_JOB_ID/$cell" \
    --result "artifacts/cluster_pilot/colocation-${SLURM_JOB_ID}-${cell}.json"
}
if [[ "${COLOCATION_PHASE:-all}" != paired ]]; then run_cell reference; fi
run_cell paired_a &
first_pid=$!
run_cell paired_b &
second_pid=$!
wait "$first_pid"
wait "$second_pid"
