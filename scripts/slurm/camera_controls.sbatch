#!/usr/bin/env bash
# One GPU, fixed five cases with two physical evaluations each. No training.
#SBATCH --job-name=plug-camera-controls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT to frozen11a591f}"
: "${CAMERA_CONTROLS_MANIFEST:?Set CAMERA_CONTROLS_MANIFEST to the submission reservation}"
: "${CAMERA_CONTROLS_HELPER:?Set CAMERA_CONTROLS_HELPER to the hash-pinned helper}"
export CAMERA_CONTROLS_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
source "${PROJECT_ROOT}/scripts/slurm/common.sh"
export PYTHONPATH="${PROJECT_ROOT}/src"
"${PROJECT_ROOT}/.venv/bin/python" - "$CAMERA_CONTROLS_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text())
for path, expected in record["launcher_files_sha256"].items():
    with Path(path).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, path
PY
exec "${PROJECT_ROOT}/.venv/bin/python" "$CAMERA_CONTROLS_HELPER" \
    --runtime "$CAMERA_CONTROLS_MANIFEST"
