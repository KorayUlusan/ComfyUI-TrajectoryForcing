#!/bin/bash
# Generate before/after EDIT pairs for N classes.
#
# The teaser's before/after beat had one pair to show -- class 213 -- which is
# half the reason the Irish setter appears in six of eight beats of a sample
# cut. A wall can be varied by picking another class; a before/after cannot,
# because an edit has to actually be RUN.
#
# Exit criteria are in scripts/edit_bank.py's docstring. The control is voiding
# and is checked before the sweep, so a drifted sampler costs seconds not
# minutes.
#
#   ./slurm/submit.sh slurm/edit_bank.sbatch                # 4-class smoke
#   CLASSES=24 ./slurm/submit.sh slurm/edit_bank.sbatch     # the bank
#
#SBATCH --job-name=tf-edit-bank
# No --partition or --qos here on purpose -- see slurm/gpu_smoke.sbatch.
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm_logs/%x-%j.out

set -euo pipefail
EXT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
EXT_DIR="$EXT"

TF_ENV_FILE="${TF_ENV_FILE:-$EXT_DIR/.env}"
if [[ -f "$TF_ENV_FILE" ]]; then set -a; . "$TF_ENV_FILE"; set +a; fi

WORK="${WORK:-$HOME}"
COMFY="${COMFY_DIR:-$WORK/ComfyUI}"
VENV="${COMFY_VENV:-$WORK/.venvs/comfyui-tf}"
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK/.cache/torch}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$WORK/.cache/jax}"
[[ -n "${TF_REPO:-}" ]] && export TF_REPO
export PYTHONPATH="$EXT:$COMFY${PYTHONPATH:+:$PYTHONPATH}"
export CLASSES="${CLASSES:-0}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node=$(hostname)  classes=${CLASSES}"
cd "$EXT"
srun "$VENV/bin/python" -u scripts/edit_bank.py
