#!/usr/bin/env bash
# Run the ComfyUI UI on a GPU node and reach it at http://localhost:PORT here.
#
#   ./serve.sh [PORT]        (default 8188)
#
# The job streams to this terminal and Ctrl-C cancels it. That is what `srun`
# gives you and `sbatch` does not: an sbatch job is detached from your shell, so
# quitting leaves an H100 allocated until the walltime runs out. The trade is
# that this job dies with this terminal, which for an interactive UI session is
# the behaviour you want -- use sbatch for anything you want to outlive it.
#
# ComfyUI binds the *compute* node, which your laptop cannot route to, so the
# script also bridges this login node's localhost:PORT to it. That means no SSH
# tunnel from here to the compute node, which on this cluster needs a password
# unless ~/.ssh/authorized_keys happens to contain your own ~/.ssh/id_*.pub.
# From your laptop you then need only the ordinary hop to the login node:
#
#   ssh -N -L PORT:localhost:PORT <user>@ferranti-login001.mlcloud.uni-tuebingen.de
#
# or nothing at all if you are on VS Code Remote, which forwards localhost ports
# for you.
set -euo pipefail

EXT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8188}"
JOB_NAME="tf-comfyui-$$"          # unique per shell, so cleanup cannot hit someone else's job
PARTITION="${TF_PARTITION:-h100-ferranti}"
QOS="${TF_QOS:-vs}"               # jumps the queue; capped at 1 job / 12 h / 4 CPU / 1 GPU
TIME_LIMIT="${TF_TIME:-08:00:00}"

command -v ncat >/dev/null || { echo "ERROR: ncat not found; it is what bridges this node to the job." >&2; exit 1; }

bridge_pid=""
cleanup() {
  trap - TERM EXIT
  echo
  echo "Shutting down: cancelling the Slurm job and closing the bridge."
  [[ -n "$bridge_pid" ]] && kill "$bridge_pid" 2>/dev/null || true
  # Belt and braces: srun releases the allocation itself on Ctrl-C, so by here
  # the job is normally already gone. This catches the case where srun was
  # killed outright and the job outlived it. By our unique job name, never by
  # a pattern -- a pattern that matches this script's own command line would
  # take out the shell running it.
  scancel --name="$JOB_NAME" --quiet 2>/dev/null || true

  # Slurm can take a few seconds to tear the allocation down, so a job still
  # listed right now is normal rather than a fault. Only say something is wrong
  # once it has had time to go.
  for _ in $(seq 1 10); do
    squeue -h -u "$USER" -n "$JOB_NAME" -o "%i" 2>/dev/null | grep -q . || break
    sleep 1
  done
  if squeue -h -u "$USER" -n "$JOB_NAME" -o "%i" 2>/dev/null | grep -q .; then
    echo "Still winding down. It should clear on its own; if it does not, run:"
    echo "    scancel --name=$JOB_NAME"
  else
    echo "Done. The GPU is released."
  fi
}
# Deliberately not trapping INT: Ctrl-C belongs to srun, which forwards it to
# ComfyUI and then releases the allocation. Cancelling from here at the same
# time would race that shutdown. The EXIT trap still fires afterwards.
trap cleanup TERM EXIT

# --- bridge: this node's localhost:PORT -> the compute node, once it is up ----
(
  node=""
  for _ in $(seq 1 900); do
    # `|| true` is not decoration: an empty squeue exits 0, but a squeue that
    # cannot reach slurmctld exits non-zero, and under `set -e` a failing
    # command substitution would kill this bridge silently while the job below
    # carried on running.
    node=$(squeue -h -u "$USER" -n "$JOB_NAME" -t R -o "%N" 2>/dev/null | head -1 || true)
    [[ -n "$node" ]] && break
    sleep 2
  done
  [[ -z "$node" ]] && exit 0
  echo ">>> job running on ${node}; bridging http://localhost:${PORT} to it." >&2
  # --keep-open so the browser's many parallel connections (and the websocket
  # ComfyUI keeps open for progress) all get through, not just the first.
  exec ncat --listen 127.0.0.1 "$PORT" --keep-open --sh-exec "ncat ${node} ${PORT}"
) &
bridge_pid=$!

cat <<EOF

  ComfyUI + Trajectory Forcing
  ────────────────────────────────────────────────────────────────────────
  What happens next:

    1. This asks Slurm for a GPU on ${PARTITION} (qos=${QOS}, up to ${TIME_LIMIT}).
       It may sit in the queue for a while -- that is normal, not a hang.
    2. Once it starts, ComfyUI boots and prints a lot of log lines.
    3. WAIT FOR THIS LINE:

           To see the GUI go to: http://0.0.0.0:${PORT}

       The port is not open until then. Ignore the 0.0.0.0 in it -- that is
       the compute node's address. Yours is:

           http://localhost:${PORT}

    4. Load a workflow from workflows/ and press Run. The first
       TF Load Pipeline takes 1-2 minutes to warm up; after that it is instant.

  Press Ctrl-C here at any time to cancel the job and free the GPU.
  ────────────────────────────────────────────────────────────────────────

EOF

# --pty gives the job a terminal so Ctrl-C reaches ComfyUI and srun then releases
# the allocation. It needs a real terminal on both ends, so fall back when either
# is redirected -- srun refuses --pty otherwise rather than degrading.
if [[ -t 0 && -t 1 ]]; then
  PTY=(--pty)
else
  PTY=(--unbuffered)
fi

srun --job-name="$JOB_NAME" \
     --partition="$PARTITION" --qos="$QOS" \
     --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=64G --time="$TIME_LIMIT" \
     "${PTY[@]}" "$EXT_DIR/run_comfyui.sh" "$PORT"
