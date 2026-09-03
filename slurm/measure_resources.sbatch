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
#SBATCH --partition=h100-ferranti
#SBATCH --qos=vs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/weka/geiger/gwb965/msc-thesis/ComfyUI-TrajectoryForcing/outputs/slurm_logs/%x-%j.out

set -euo pipefail
WORK="${WORK:-/weka/geiger/gwb965}"
EXT="$WORK/msc-thesis/ComfyUI-TrajectoryForcing"
export COMFY_DIR="${COMFY_DIR:-$WORK/ComfyUI}"
# folder_paths is ComfyUI's, and this script calls it directly rather than
# through a running server, so ComfyUI has to be importable.
export PYTHONPATH="$EXT:$COMFY_DIR${PYTHONPATH:+:$PYTHONPATH}"
VENV="${COMFY_VENV:-$WORK/.venvs/comfyui-tf}"

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK/.cache/torch}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$WORK/.cache/jax}"
export TF_REPO="$WORK/msc-thesis/TrajectoryForcing"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node=$(hostname)"

cd "$EXT"
# A port derived from the job id, so two of these never collide on one node.
srun "$VENV/bin/python" -u scripts/measure_resources.py
