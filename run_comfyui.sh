#!/usr/bin/env bash
# Launch ComfyUI with the JAX/PyTorch settings Trajectory Forcing needs.
#
#   ./run_comfyui.sh [PORT]        (default 8188, ComfyUI's own default)
#
# The extension sets the same JAX variables itself when it loads, so ComfyUI
# started any other way still works. This script exists because setting them
# here is strictly earlier -- before ComfyUI imports anything -- which is the
# only ordering that cannot be broken by another custom node importing jax
# first. Anything already exported wins, so the two never disagree.
set -euo pipefail

EXT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Local overrides, if you made one: `cp .env.example .env`. Gitignored, so your
# paths never reach a commit. .env.example writes every line as
# `KEY="${KEY:-default}"`, so a value exported on the command line still wins.
# TF_ENV_FILE points somewhere else -- one file per cluster, say.
TF_ENV_FILE="${TF_ENV_FILE:-$EXT_DIR/.env}"
if [[ -f "$TF_ENV_FILE" ]]; then
  set -a; . "$TF_ENV_FILE"; set +a
fi

WORK="${WORK:-$HOME}"
COMFY_DIR="${COMFY_DIR:-$WORK/ComfyUI}"
VENV="${COMFY_VENV:-$WORK/.venvs/comfyui-tf}"
PORT="${1:-8188}"

[[ -d "$COMFY_DIR" ]] || { echo "ERROR: no ComfyUI at $COMFY_DIR (set COMFY_DIR)" >&2; exit 1; }
[[ -x "$VENV/bin/python" ]] || { echo "ERROR: no venv at $VENV (set COMFY_VENV)" >&2; exit 1; }

# --- GPU selection ---------------------------------------------------------
# Under Slurm the allocated device is already in CUDA_VISIBLE_DEVICES; GPU=<n>
# overrides it for a manual run.
if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if ! "$VENV/bin/python" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "ERROR: CUDA is not available -- TF would run on CPU, which is unusably slow." >&2
  exit 1
fi
echo "CUDA OK: $("$VENV/bin/python" -c 'import torch;print(torch.cuda.get_device_name(0))')"

# --- JAX / torch sharing one card ------------------------------------------
# From editing_env/run.sh, which already runs this exact pair of frameworks in
# one process. PREALLOCATE=false is the important one: left at its default JAX
# claims ~75% of the card the moment it initialises, and comfy.model_management
# cannot see that allocation, so it keeps loading torch models into VRAM that is
# already gone.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$WORK/.cache/jax}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# A hard ceiling on JAX's share, for a workflow that also loads big torch models.
# MEM_FRACTION only sizes the *preallocated* block, so asking for a ceiling means
# turning preallocation back on -- growth-on-demand and a cap are alternatives,
# not a pair. Unset by default: TF's L/16 model plus the ViT-XL decoder is a few
# GB on an 80 GB card, so the cap is only worth it when something else is big.
if [[ -n "${TF_XLA_MEM_FRACTION:-}" ]]; then
  export XLA_PYTHON_CLIENT_PREALLOCATE=true
  export XLA_PYTHON_CLIENT_MEM_FRACTION="$TF_XLA_MEM_FRACTION"
  # Tell ComfyUI's allocator to stay out of JAX's block, in the same units it uses.
  RESERVE_GB="${TF_RESERVE_VRAM:-8}"
  COMFY_EXTRA=(--reserve-vram "$RESERVE_GB")
  echo "JAX capped at ${TF_XLA_MEM_FRACTION} of the card; ComfyUI reserving ${RESERVE_GB} GB"
else
  COMFY_EXTRA=()
fi

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$WORK/.cache/torch}"
export TF_REPO="${TF_REPO:-$(cd "$EXT_DIR/.." && pwd)/TrajectoryForcing}"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

# --- the custom_nodes symlink ----------------------------------------------
LINK="$COMFY_DIR/custom_nodes/$(basename "$EXT_DIR")"
if [[ ! -e "$LINK" ]]; then
  ln -s "$EXT_DIR" "$LINK"
  echo "linked $LINK -> $EXT_DIR"
fi

# --- keep a log ------------------------------------------------------------
# Without this an interactive session leaves no evidence: run_comfyui_slurm.sh streams to a
# terminal nobody keeps, so "I pressed Run and nothing happened" has nothing to
# read afterwards. ComfyUI's own `--verbose LEVEL FILE` writes the file itself,
# which matters -- piping through `tee` would make stdout a pipe, and `srun
# --pty` needs a real terminal on both ends or Ctrl-C stops reaching ComfyUI.
LOG_DIR="${TF_LOG_DIR:-$EXT_DIR/outputs/comfyui_logs}"
mkdir -p "$LOG_DIR"
# Keep the most recent $KEEP_LOGS of a family, quietly and without ever failing.
KEEP_LOGS=20
prune_logs() {
  local stem="$1" old
  old=$(ls -1t "$LOG_DIR/$stem"-*.log 2>/dev/null | tail -n +$((KEEP_LOGS + 1)) || true)
  [[ -n "$old" ]] && printf '%s\n' "$old" | xargs -r rm -f
  return 0
}

SERVER_LOG="$LOG_DIR/comfyui-$(date +%Y%m%d-%H%M%S)-$$.log"
# Keep the last 20; these are a few hundred KB each and nobody prunes by hand.
# `|| true` is load-bearing: an empty glob makes `ls` exit 2, `2>/dev/null`
# hides the message but not the status, and `set -o pipefail` + `set -e` then
# kill the script before it prints anything. That is exactly what happened --
# the logging added to diagnose a silent failure caused one.
prune_logs "comfyui"

echo "TrajectoryForcing repo: $TF_REPO"
echo "ComfyUI on port $PORT; the first TF Load Pipeline warms up for 1-2 minutes."
echo "server log: $SERVER_LOG"
cd "$COMFY_DIR"
exec "$VENV/bin/python" -u main.py --listen 0.0.0.0 --port "$PORT" \
     --verbose INFO "$SERVER_LOG" "${COMFY_EXTRA[@]}" "${@:2}"
