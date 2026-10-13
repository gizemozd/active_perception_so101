#!/usr/bin/env bash
# CPU-only follow-up: refresh results even if the training job fails.
#SBATCH --job-name=plug-recipe-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
export TRAINING_REPORT_ROOT="$1"
snapshot="$2"
export MPLCONFIGDIR="/tmp/plug-recipe-report-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=2
cd "$snapshot"
sha256sum -c sha256sum.txt
"$TRAINING_REPORT_ROOT/.venv/bin/python" "$snapshot/build_plug_repair_report.py"
"$TRAINING_REPORT_ROOT/.venv/bin/python" "$snapshot/build_training_results_report.py"
"$TRAINING_REPORT_ROOT/.venv/bin/python" "$snapshot/validate_training_results_report.py"
