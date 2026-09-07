#!/bin/bash
# Generate the four decoded levels for N ImageNet classes, as a 256px tile bank.
#
# Exit criteria are in scripts/class_grid.py's docstring, including the control
# (class 213 / seed 592 reproduces the 2026-09-04 gpu_smoke decodes bit-for-bit);
# if the control fails the run is void and no timing or tile from it can be cited.
#
#   ./slurm/submit.sh slurm/class_grid.sbatch                # 4-class smoke arm
#   CLASSES=28 ./slurm/submit.sh slurm/class_grid.sbatch     # fills a 4K frame
#   CLASSES=1000 EXTRA_SEEDS='7,101,4242' ./slurm/submit.sh slurm/class_grid.sbatch
#                                                          # 4000 trajectories
#
# Run the smoke arm and read its timing before sizing anything larger.
#
#SBATCH --job-name=tf-class-grid
# No --partition or --qos here on purpose -- see slurm/gpu_smoke.sbatch.
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
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
export EXTRA_SEEDS="${EXTRA_SEEDS:-}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-<unset>}  classes=${CLASSES}  extra_seeds=${EXTRA_SEEDS:-none}"

cd "$EXT"
srun "$VENV/bin/python" -u scripts/class_grid.py
