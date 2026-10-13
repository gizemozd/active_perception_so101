#!/usr/bin/env bash
# Refresh report even after a pilot failure or scheduler time limit.
#SBATCH --job-name=transfer-report
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
export TRANSFER_REPORT_ROOT="$1"
snapshot="$2"
export PYTHONPATH="$3/src"
export MPLCONFIGDIR="/tmp/transfer-report-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=2
cd "$snapshot"
sha256sum --quiet -c sha256sum.txt
"$TRANSFER_REPORT_ROOT/.venv/bin/python" - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ['TRANSFER_REPORT_ROOT']) / 'artifacts/transfer_repair/pilot-runtime.json'
if path.exists():
    value = json.loads(path.read_text())
    if value['status'] == 'running':
        value['status'] = 'incomplete'
        value['error'] = 'The scheduler job ended before the launcher recorded completion. Inspect the Slurm logs.'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)
PY
"$TRANSFER_REPORT_ROOT/.venv/bin/python" build_transfer_repair_report.py
