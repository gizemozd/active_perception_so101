#!/usr/bin/env bash
# Prepared only. Submit after final fresh751 audit fails the allvariant gate.
# No dependency auto-launch: root supplies final checkpoint/reference and hashes.
#SBATCH --job-name=plug-target-diagnostic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT:?Set PROJECT_ROOT to frozen11a591f}/scripts/slurm/common.sh"
: "${DATA_ROOT:?Set DATA_ROOT to the results checkout}"
: "${TARGET_CHECKPOINT:?Set explicit final fresh model_750.pt}"
: "${TARGET_REFERENCE_REPORT:?Set final fresh512physical evaluation report}"
: "${TARGET_CHECKPOINT_SHA256:?Freeze checkpoint SHA256 before submission}"
: "${TARGET_HELPER_SHA256:?Freeze helper SHA256 before submission}"
target_helper="$DATA_ROOT/scripts/audit_plug_actor_targets.py"
"$DATA_ROOT/.venv/bin/python" - "$target_helper" "$TARGET_HELPER_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == sys.argv[2], "Helper source differs"
PY
export PYTHONPATH="$PROJECT_ROOT/src"
args=("$target_helper" --device cuda:0 --checkpoint "$TARGET_CHECKPOINT"
  --expected-checkpoint-sha256 "$TARGET_CHECKPOINT_SHA256"
  --reference-report "$TARGET_REFERENCE_REPORT"
  --output "${TARGET_OUTPUT:-$DATA_ROOT/artifacts/plug_analysis/target_diagnostic/report.json}")
if [[ "${TARGET_PROBE:-0}" == 1 ]]; then args+=(--probe); fi
run_python "${args[@]}"
