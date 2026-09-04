#!/usr/bin/env bash
# run_comfyui.sh, but on a Slurm GPU node: allocates one, runs ComfyUI there,
# and gets you to it at http://localhost:PORT on this login node.
#
#   ./run_comfyui_slurm.sh [PORT]        (default 8188)
#
# Use ./run_comfyui.sh instead if you are already on a machine with a GPU.
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
#   ssh -N -L PORT:localhost:PORT <user>@<your-login-node>
#
# or nothing at all if you are on VS Code Remote, which forwards localhost ports
# for you.
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

PORT="${1:-8188}"
JOB_NAME="tf-comfyui-$$"          # unique per shell, so cleanup cannot hit someone else's job
# No defaults: a partition or QOS name is a property of *your* cluster, and a
# wrong one fails with a Slurm error that does not say which of the two it was.
# Unset means "do not pass the flag", so Slurm applies the cluster default --
# which is right on a cluster with a sensible one, and on a cluster without you
# want to have chosen anyway. Set them in .env; see .env.example.
# `sinfo -s` lists partitions, `sacctmgr show qos format=name` lists QOS names.
PARTITION="${TF_PARTITION:-}"
QOS="${TF_QOS:-}"
TIME_LIMIT="${TF_TIME:-08:00:00}"

# Built as an array so an unset value contributes no argument at all, rather
# than an empty `--partition=` that Slurm rejects.
SLURM_OPTS=()
[[ -n "$PARTITION" ]] && SLURM_OPTS+=(--partition="$PARTITION")
[[ -n "$QOS" ]] && SLURM_OPTS+=(--qos="$QOS")

command -v ncat >/dev/null || { echo "ERROR: ncat not found; it is what bridges this node to the job." >&2; exit 1; }

# --- clear our own leftovers, then find a port that is actually free ---------
# `ncat --keep-open` forks a child per connection and each fork inherits the
# listening socket, so killing the parent alone leaves the port bound by
# orphans -- which is what used to happen, and why a later run failed with
# "Address already in use" pointing at a compute node whose job had long ended.
# The bridge now runs in its own process group and is killed by group; this
# sweep clears anything left behind by a version that did not.
sweep_stale_bridges() {
  local pid cmdline node stale=()
  for pid in $(pgrep -u "$USER" -x ncat 2>/dev/null || true); do
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    # The node comes out of the bridge's own command line -- `--sh-exec ncat
    # <node> <port>` -- rather than a hostname pattern. This used to grep for
    # 'mlcbm[0-9]+', so on any cluster whose nodes are named differently the
    # sweep silently matched nothing and the leak it exists to clear came back
    # with no error to explain it.
    node=$(sed -nE 's/.*--sh-exec[[:space:]]+ncat[[:space:]]+([^[:space:]]+).*/\1/p' <<<"$cmdline" | head -1)
    # Only ours, and only when no job of ours is still on that node.
    [[ -z "$node" ]] && continue
    if ! squeue -h -u "$USER" -t R -o "%N" 2>/dev/null | grep -qx "$node"; then
      stale+=("$pid")
    fi
  done
  if (( ${#stale[@]} )); then
    echo "Clearing ${#stale[@]} leftover bridge process(es) from an earlier session."
    kill "${stale[@]}" 2>/dev/null || true
    sleep 1
    kill -9 "${stale[@]}" 2>/dev/null || true
  fi
}

port_free() { ! ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }

sweep_stale_bridges
if ! port_free "$PORT"; then
  original="$PORT"
  while ! port_free "$PORT" && (( PORT < original + 20 )); do PORT=$((PORT + 1)); done
  if ! port_free "$PORT"; then
    echo "ERROR: ports $original..$PORT are all in use on this node." >&2
    echo "       Something else is listening -- check: ss -ltnp | grep $original" >&2
    exit 1
  fi
  echo "Port $original is in use here (VS Code forwards ports it sees, and it may be one)."
  echo "Using $PORT instead."
fi

# Both logs land in one directory so a report of "it did nothing" has one place
# to look. Under /weka rather than /tmp, because /tmp is node-local: the bridge
# runs here on the login node and ComfyUI runs on the compute node, and only a
# shared filesystem puts the two halves of a session side by side.
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

# `|| true` is load-bearing: an empty glob makes `ls` exit 2, `2>/dev/null`
# hides the message but not the status, and `set -o pipefail` + `set -e` then
# kill the script before it prints anything. That is exactly what happened --
# the logging added to diagnose a silent failure caused one.
prune_logs "bridge"
BRIDGE_LOG="$LOG_DIR/bridge-$(date +%Y%m%d-%H%M%S)-$$.log"
bridge_pid=""
cleanup() {
  trap - TERM EXIT
  echo
  echo "Shutting down: cancelling the Slurm job and closing the bridge."
  # By process group, not PID: ncat's per-connection forks each hold the
  # listening socket, so killing the leader alone leaves the port bound.
  [[ -n "$bridge_pid" ]] && kill -- -"$bridge_pid" 2>/dev/null || true
  # Belt and braces: srun releases the allocation itself on Ctrl-C, so by here
  # the job is normally already gone. This catches the case where srun was
  # killed outright and the job outlived it. By our unique job name, never by
  # a pattern -- a pattern that matches this script's own command line would
  # take out the shell running it.
  scancel --name="$JOB_NAME" --quiet 2>/dev/null || true
  # A bridge that never bound leaves the printed URL dead with no other sign.
  if [[ -s "$BRIDGE_LOG" ]] && grep -qi "QUITTING\|Address already in use" "$BRIDGE_LOG"; then
    echo "NOTE: the port bridge failed, so http://localhost:${PORT} was never live:"
    sed 's/^/      /' "$BRIDGE_LOG"
  fi
  # Deliberately not deleted. This used to be `rm -f`, which meant a session
  # where the UI misbehaved left nothing at all to look at afterwards -- the
  # terminal stream is gone the moment the window is, and the server's own log
  # lives on the compute node's view of $EXT_DIR. Say where both are.
  echo
  echo "Logs from this session, if anything looked wrong:"
  echo "    bridge: $BRIDGE_LOG"
  echo "    server: $(ls -1t "${TF_LOG_DIR:-$EXT_DIR/outputs/comfyui_logs}"/comfyui-*.log 2>/dev/null | head -1 || echo '<none written>')"

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
run_bridge() {
  local node=""
  for _ in $(seq 1 900); do
    # `|| true` is not decoration: an empty squeue exits 0, but a squeue that
    # cannot reach slurmctld exits non-zero, and under `set -e` a failing
    # command substitution would kill this bridge silently while the job below
    # carried on running.
    node=$(squeue -h -u "$USER" -n "$JOB_NAME" -t R -o "%N" 2>/dev/null | head -1 || true)
    [[ -n "$node" ]] && break
    sleep 2
  done
  [[ -z "$node" ]] && return 0
  echo "bridging localhost:${PORT} -> ${node}:${PORT}"
  # --keep-open so the browser's many parallel connections (and the websocket
  # ComfyUI keeps open for progress) all get through, not just the first.
  exec ncat --listen 127.0.0.1 "$PORT" --keep-open --sh-exec "ncat ${node} ${PORT}"
}
export -f run_bridge
export JOB_NAME PORT

# setsid: its own process group, so cleanup can kill ncat's per-connection forks
# as a group. Output goes to a file rather than the terminal -- srun --pty puts
# the terminal in raw mode, and a background writer into it produces the
# stair-stepped, half-overwritten lines this used to print.
setsid bash -c run_bridge >"$BRIDGE_LOG" 2>&1 &
bridge_pid=$!

cat <<EOF

  ComfyUI + Trajectory Forcing
  ────────────────────────────────────────────────────────────────────────
  What happens next:

    1. This asks Slurm for a GPU on ${PARTITION:-<cluster default>} (qos=${QOS:-<cluster default>}, up to ${TIME_LIMIT}).
       It may sit in the queue for a while -- that is normal, not a hang.
    2. Once it starts, ComfyUI boots and prints a lot of log lines.
    3. WAIT FOR THIS LINE:

           To see the GUI go to: http://0.0.0.0:${PORT}

       The port is not open until then. Ignore the 0.0.0.0 in it -- that is
       the compute node's address. Yours is:

           http://localhost:${PORT}

    4. Load a workflow from example_workflows/ and press Run. The first
       TF Load Pipeline takes 1-2 minutes to warm up; after that it is instant.

  Press Ctrl-C here at any time to cancel the job and free the GPU.
  (bridge log: ${BRIDGE_LOG})
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
     "${SLURM_OPTS[@]}" \
     --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=64G --time="$TIME_LIMIT" \
     "${PTY[@]}" "$EXT_DIR/run_comfyui.sh" "$PORT"
