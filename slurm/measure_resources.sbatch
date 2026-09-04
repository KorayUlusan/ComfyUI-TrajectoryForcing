#!/bin/bash
# Run the example workflows through a real ComfyUI server on a GPU.
#
# Exit criteria are in scripts/server_smoke.py's docstring. This is the test
# that covers what gpu_smoke.py cannot: ComfyUI's own execution engine, its
# validation of the custom socket types, and whether workflows/api/*.json are
# executable rather than merely well-formed.
#
#   sbatch slurm/server_smoke.sbatch
#
#SBATCH --job-name=tf-comfy-resources
# No --partition or --qos here on purpose: they differ on every cluster and
# an #SBATCH line cannot read a variable. `./slurm/submit.sh` passes yours
# from .env on the command line, where they override anything in this file.
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
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
export COMFY_DIR="${COMFY_DIR:-$WORK/ComfyUI}"
# folder_paths is ComfyUI's, and this script calls it directly rather than
# through a running server, so ComfyUI has to be importable.
export PYTHONPATH="$EXT:$COMFY_DIR${PYTHONPATH:+:$PYTHONPATH}"
VENV="${COMFY_VENV:-$WORK/.venvs/comfyui-tf}"

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK/.cache/torch}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$WORK/.cache/jax}"
# Only if you set it; otherwise tf_nodes/locate.py finds or fetches it.
[[ -n "${TF_REPO:-}" ]] && export TF_REPO
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node=$(hostname)"

cd "$EXT"
# A port derived from the job id, so two of these never collide on one node.
srun "$VENV/bin/python" -u scripts/measure_resources.py
