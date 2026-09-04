#!/usr/bin/env bash
# Build the venv that runs ComfyUI and Trajectory Forcing in one process.
#
#   bash env/setup.sh                 (~10 min, ~11 GB)
#
# Not idempotent by design: it creates the venv from scratch, because a
# half-resolved mix of the two dependency sets is much harder to diagnose than a
# rebuild. Delete $VENV first to redo it.
#
# Order is the whole point. TrajectoryForcing's pinned JAX stack goes in first,
# so that when ComfyUI's unpinned `torch` / `torchvision` lines are resolved they
# are already satisfied and pip does not pull a different CUDA build underneath
# jax. See env/requirements.txt for the one pin that had to move, and why.
set -euo pipefail

EXT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Local overrides, if you made one: `cp .env.example .env`. Gitignored, so your
# paths never reach a commit. .env.example writes every line as
# `KEY="${KEY:-default}"`, so a value exported on the command line still wins.
# TF_ENV_FILE points somewhere else -- one file per cluster, say.
TF_ENV_FILE="${TF_ENV_FILE:-$EXT_DIR/.env}"
if [[ -f "$TF_ENV_FILE" ]]; then
  set -a; . "$TF_ENV_FILE"; set +a
fi

WORK="${WORK:-$HOME}"
VENV="${COMFY_VENV:-$WORK/.venvs/comfyui-tf}"
COMFY="${COMFY_DIR:-$WORK/ComfyUI}"
PY="${PYTHON:-/usr/bin/python3.11}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK/.cache/pip}"

[[ -d "$COMFY" ]] || { echo "ERROR: no ComfyUI at $COMFY (set COMFY_DIR)" >&2; exit 1; }
[[ -e "$VENV" ]] && { echo "ERROR: $VENV already exists; remove it to rebuild." >&2; exit 1; }

"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel setuptools

echo "=== [1/4] TrajectoryForcing's JAX stack ==="
"$VENV/bin/pip" install \
  "jax[cuda12]==0.4.36" "flax==0.10.4" "optax==0.2.5" "orbax-checkpoint==0.11.0" \
  "chex==0.1.87" "ml_dtypes==0.5.1" "tensorstore==0.1.76" "ml_collections==1.1.0"

echo "=== [2/4] torch (cu128, above TF's own pin -- see env/requirements.txt) ==="
"$VENV/bin/pip" install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== [3/4] TF inference extras ==="
"$VENV/bin/pip" install \
  "transformers==5.3.0" huggingface_hub PyYAML "numpy>=2.2" scipy scikit-learn \
  pillow einops timm pytest \
  "wandb==0.22.0"   # not optional: utils/logging_util.py imports it at module scope

echo "=== [4/4] ComfyUI ==="
"$VENV/bin/pip" install -r "$COMFY/requirements.txt"

echo "=== check ==="
"$VENV/bin/pip" check
PYTHONPATH="$COMFY" "$VENV/bin/python" - <<'PY'
import importlib
for name in ("jax", "flax", "torch", "numpy", "transformers"):
    print(f"{name:14s} {importlib.import_module(name).__version__}")
# the pairing that actually breaks: ComfyUI's kernel library against torch
import comfy.quant_ops  # noqa: F401
print("comfy.quant_ops   imports cleanly")
PY

echo
echo "Done: $VENV"
echo "Next: ./run_comfyui.sh   (or sbatch slurm/comfyui.sbatch on a GPU node)"
