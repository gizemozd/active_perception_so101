#!/usr/bin/env bash
# Run a concrete prepared manifest; validation is a separate short allocation.
#SBATCH --job-name=plug-recipe-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
manifest="$1"
stage="$2"
python="$3"
helper="$4"
exec "$python" "$helper" --manifest "$manifest" --stage "$stage"
