#!/usr/bin/env bash
#SBATCH --job-name=vision-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
run_python -m pytest -q tests/test_mjlab.py tests/test_policy.py tests/test_plug.py tests/test_cli.py tests/test_evaluation.py
run_python -m active_perception_arms.benchmark --task plug --num-envs 32 --steps 25 --actor --wandb \
 --output "artifacts/gpu-smoke-${SLURM_JOB_ID:-local}.json"

run_python -m active_perception_arms.plug_audit --backend warp --device cuda:0 --output "artifacts/cluster_pilot/validation-${SLURM_JOB_ID}"
run_python -m active_perception_arms.gpu_validate
run_python -c 'import json,os; from pathlib import Path; r=json.loads((Path("artifacts/cluster_pilot") / ("validation-"+os.environ["SLURM_JOB_ID"]) / "warp/warp_report.json").read_text()); assert all(r["scripted_success"]) and all(r["final_success"]), r'
