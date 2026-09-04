#!/bin/bash
# End-to-end smoke test of the node graph against the real TF model.
#
# Exit criteria are written down in scripts/gpu_smoke.py's docstring, including
# the control (same class + seed reproduces the trajectory bit-for-bit); if the
# control fails the run is void and no other result from it can be cited.
#
#   ./slurm/submit.sh slurm/gpu_smoke.sbatch
#
#SBATCH --job-name=tf-comfy-smoke
# No --partition or --qos here on purpose: they differ on every cluster and
# an #SBATCH line cannot read a variable. `./slurm/submit.sh` passes yours
# from .env on the command line, where they override anything in this file.
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
# Relative, so this is the one #SBATCH line that needs no editing: Slurm
# resolves it against the submission directory. Submit from the repo root.
#SBATCH --output=outputs/slurm_logs/%x-%j.out

set -euo pipefail
# Where sbatch was run from, which the --output line above also assumes.
EXT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
EXT_DIR="$EXT"

# Local overrides, if you made one: `cp .env.example .env`. Gitignored, so your
# paths never reach a commit. .env.example writes every line as
# `KEY="${KEY:-default}"`, so a value exported on the command line still wins.
# TF_ENV_FILE points somewhere else -- one file per cluster, say.
TF_ENV_FILE="${TF_ENV_FILE:-$EXT_DIR/.env}"
if [[ -f "$TF_ENV_FILE" ]]; then
  set -a; . "$TF_ENV_FILE"; set +a
fi

WORK="${WORK:-$HOME}"
COMFY="${COMFY_DIR:-$WORK/ComfyUI}"
VENV="${COMFY_VENV:-$WORK/.venvs/comfyui-tf}"

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK/.cache/torch}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$WORK/.cache/jax}"
# Only if you set it; otherwise tf_nodes/locate.py finds or fetches it.
[[ -n "${TF_REPO:-}" ]] && export TF_REPO
# ComfyUI's root has to be importable: the nodes import comfy_api, and the model
# folder comes from folder_paths.
export PYTHONPATH="$EXT:$COMFY${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-<unset>}"

cd "$EXT"
srun "$VENV/bin/python" -u scripts/gpu_smoke.py
