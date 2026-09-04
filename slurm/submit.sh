#!/usr/bin/env bash
# Submit one of the slurm/*.sbatch jobs with your cluster's partition and QOS.
#
#   ./slurm/submit.sh slurm/gpu_smoke.sbatch
#   ./slurm/submit.sh slurm/server_smoke.sbatch --time=01:00:00   # extra sbatch flags pass through
#
# Why this exists: `#SBATCH` lines are comments to bash, so they cannot read a
# variable, which makes partition and QOS -- the two settings that are different
# on every cluster -- the one thing a .sbatch file cannot parameterise. Naming
# them there would hardcode this machine into the repo; leaving them out entirely
# would send the job to whatever partition is default, which on many clusters has
# no GPU. So they come from .env and go on the sbatch command line, where they
# override anything in the file.
#
# Plain `sbatch slurm/gpu_smoke.sbatch` still works if your cluster's default
# partition has GPUs, or if you pass --partition yourself.
set -euo pipefail

EXT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TF_ENV_FILE="${TF_ENV_FILE:-$EXT_DIR/.env}"
if [[ -f "$TF_ENV_FILE" ]]; then
  set -a; . "$TF_ENV_FILE"; set +a
fi

[[ $# -ge 1 ]] || { echo "usage: $0 slurm/<job>.sbatch [extra sbatch flags]" >&2; exit 2; }

# An unset value contributes no argument, rather than an empty `--partition=`
# that Slurm rejects outright.
OPTS=()
[[ -n "${TF_PARTITION:-}" ]] && OPTS+=(--partition="$TF_PARTITION")
[[ -n "${TF_QOS:-}" ]] && OPTS+=(--qos="$TF_QOS")

# From the repo root, because every job's `#SBATCH --output=` is relative and
# Slurm resolves it against the submission directory.
cd "$EXT_DIR"
echo "sbatch ${OPTS[*]:-<cluster defaults>} $*"
exec sbatch "${OPTS[@]}" "$@"
