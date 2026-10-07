#!/usr/bin/env bash
set -euo pipefail
# Run once in the project root after loading your site's Python/CUDA environment.
command -v uv >/dev/null
# Triton compiles a small C extension; a runtime-only system Python is insufficient.
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c 'import pathlib,sysconfig; assert (pathlib.Path(sysconfig.get_path("include"))/"Python.h").is_file()'; then
  uv venv --allow-existing --managed-python --python 3.12 .venv
fi
uv sync --locked
mkdir -p logs/slurm artifacts
uv run --no-sync python -c 'import torch, mujoco, mjlab, warp; print("torch", torch.__version__, "MuJoCo", mujoco.__version__, "CUDA available", torch.cuda.is_available())'
