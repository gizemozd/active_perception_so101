#!/usr/bin/env bash
set -euo pipefail
# Run once in the project root after loading your site's Python/CUDA environment.
command -v uv >/dev/null
uv sync --locked
mkdir -p logs/slurm artifacts
uv run --no-sync python -c 'import torch, mujoco, mjlab, warp; print("torch", torch.__version__, "MuJoCo", mujoco.__version__, "CUDA available", torch.cuda.is_available())'
