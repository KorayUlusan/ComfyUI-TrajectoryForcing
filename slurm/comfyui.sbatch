#!/bin/bash
# Serve the ComfyUI UI from a GPU node, reachable over an SSH tunnel.
#
#   sbatch slurm/comfyui.sbatch
#   tail -f outputs/slurm_logs/tf-comfyui-<jobid>.out   # prints the tunnel command
#
# --qos=vs jumps the queue but is capped at one job, 12 h, 4 CPUs and 1 GPU --
# the right trade for an interactive session, and the reason the resource
# requests below are small. Drop the line for a longer or larger run.
#
#SBATCH --job-name=tf-comfyui
#SBATCH --partition=h100-ferranti
#SBATCH --qos=vs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/weka/geiger/gwb965/msc-thesis/ComfyUI-TrajectoryForcing/outputs/slurm_logs/%x-%j.out

set -euo pipefail
WORK="${WORK:-/weka/geiger/gwb965}"
EXT="$WORK/msc-thesis/ComfyUI-TrajectoryForcing"

# A port derived from the job id, so two sessions never collide on one node.
PORT=$((8300 + SLURM_JOB_ID % 200))
NODE=$(hostname)

cat <<EOF

  ComfyUI is starting on ${NODE}:${PORT}.

  From your laptop:
      ssh -N -L ${PORT}:${NODE}:${PORT} <user>@ferranti-login001.mlcloud.uni-tuebingen.de
      open http://localhost:${PORT}

  From this login node:
      ssh -N -L ${PORT}:${NODE}:${PORT} ${NODE}

  Then load a workflow from ${EXT}/workflows/.
  The first TF Load Pipeline warms up for a minute or two; the node stays
  highlighted until it finishes.

EOF

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$EXT"
exec ./run_comfyui.sh "$PORT"
