#!/bin/bash
# Generate the four PCA token mosaics for N ImageNet classes, as a 512px bank.
#
# The latent-space companion to slurm/class_grid.sbatch. It skips the RAE
# decoder, which is the expensive half, so it is cheaper per class than the
# decode bank -- read the smoke arm's timing before sizing anything larger.
#
# Exit criteria are in scripts/pca_bank.py's docstring. TWO of them are voiding:
# the control class must decode bit-for-bit to the 2026-09-04 gpu_smoke frames
# AND reproduce that run's PCA mosaics. The first is what makes a PCA tile and a
# decode tile of the same class the same trajectory, which several shots pair
# on screen.
#
#   ./slurm/submit.sh slurm/pca_bank.sbatch                # 4-class smoke arm
#   CLASSES=1000 ./slurm/submit.sh slurm/pca_bank.sbatch   # the full bank
#
#SBATCH --job-name=tf-pca-bank
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
if [[ -f "$TF_ENV_FILE" ]]; then
  set -a; . "$TF_ENV_FILE"; set +a
fi

WORK="${WORK:-$HOME}"
COMFY="${COMFY_DIR:-$WORK/ComfyUI}"
VENV="${COMFY_VENV:-$WORK/.venvs/comfyui-tf}"

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK/.cache/torch}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$WORK/.cache/jax}"
[[ -n "${TF_REPO:-}" ]] && export TF_REPO
export PYTHONPATH="$EXT:$COMFY${PYTHONPATH:+:$PYTHONPATH}"

# Passed through to the script; unset means the 4-class smoke arm.
export CLASSES="${CLASSES:-0}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-<unset>}  classes=${CLASSES}"

cd "$EXT"
srun "$VENV/bin/python" -u scripts/pca_bank.py
