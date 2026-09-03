#!/bin/bash
# End-to-end smoke test of the node graph against the real TF model.
#
# Exit criteria are written down in scripts/gpu_smoke.py's docstring, including
# the control (same class + seed reproduces the trajectory bit-for-bit); if the
# control fails the run is void and no other result from it can be cited.
#
#   sbatch slurm/gpu_smoke.sbatch
#
#SBATCH --job-name=tf-comfy-smoke
#SBATCH --partition=h100-ferranti
#SBATCH --qos=vs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/weka/geiger/gwb965/msc-thesis/ComfyUI-TrajectoryForcing/outputs/slurm_logs/%x-%j.out

set -euo pipefail
WORK="${WORK:-/weka/geiger/gwb965}"
EXT="$WORK/msc-thesis/ComfyUI-TrajectoryForcing"
COMFY="${COMFY_DIR:-$WORK/ComfyUI}"
VENV="${COMFY_VENV:-$WORK/.venvs/comfyui-tf}"

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK/.cache/torch}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$WORK/.cache/jax}"
export TF_REPO="$WORK/msc-thesis/TrajectoryForcing"
# ComfyUI's root has to be importable: the nodes import comfy_api, and the model
# folder comes from folder_paths.
export PYTHONPATH="$EXT:$COMFY${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-<unset>}"

cd "$EXT"
srun "$VENV/bin/python" -u scripts/gpu_smoke.py
