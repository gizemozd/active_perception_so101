#!/usr/bin/env bash
# Prepared only:512 first episodes, unchanged privileged controller, actual Warp physics.
# Source11a591f stays frozen; DATA_ROOT owns this audited helper and its output.
#SBATCH --job-name=plug-physical-oracle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT:?Set PROJECT_ROOT to frozen11a591f}/scripts/slurm/common.sh"
: "${DATA_ROOT:?Set DATA_ROOT to the results checkout}"
: "${AUDIT_SCRIPT_SHA256:?Set the frozen helper SHA256 before submission}"
audit_script="$DATA_ROOT/scripts/audit_randomized_plug_feasibility_gpu.py"
"$DATA_ROOT/.venv/bin/python" - "$audit_script" "$AUDIT_SCRIPT_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == sys.argv[2], "Helper source differs"
PY
export PYTHONPATH="$PROJECT_ROOT/src"
run_python "$audit_script" --device cuda:0 \
  --reference-report "$DATA_ROOT/artifacts/plug_analysis/corrected-wrist_static-s0.json" \
  --output "${AUDIT_OUTPUT:-$DATA_ROOT/artifacts/plug_analysis/randomized_feasibility_gpu.json}"
