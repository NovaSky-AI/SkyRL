#!/usr/bin/env bash
#SBATCH --job-name=r2egym-ta-32b
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=176
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Submit with, for example:
#   MODEL_PATH=/path/to/Qwen3-32B DATA_ROOT=$HOME/data/harbor \
#     sbatch --partition=<partition> examples/train/thunder_agent/scripts/r2egym_32b/run_sbatch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# run_stages.sh is the only wrapper this sbatch entrypoint calls. It starts the
# rollout servers and trainer driver through script-local helpers on the right
# Slurm nodes.
WRAPPER="$SCRIPT_DIR/run_stages.sh"

cd "$REPO_ROOT"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "This script must run inside sbatch." >&2
  exit 1
fi

mapfile -t ALLOC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#ALLOC_NODES[@]}" -lt 5 ]; then
  echo "This recipe needs 5 allocated nodes, got ${#ALLOC_NODES[@]}." >&2
  exit 1
fi

export MERGED_JOB_ID="$SLURM_JOB_ID"
export MERGED_NODE="${MERGED_NODE:-${ALLOC_NODES[0]}}"
export ROLLOUT_JOB_ID="${ROLLOUT_JOB_ID:-$SLURM_JOB_ID}"
export ROLLOUT_NODE="${ROLLOUT_NODE:-${ALLOC_NODES[0]}}"
export TRAINER_NODE_SPECS="${TRAINER_NODE_SPECS:-${ALLOC_NODES[1]}:$SLURM_JOB_ID,${ALLOC_NODES[2]}:$SLURM_JOB_ID,${ALLOC_NODES[3]}:$SLURM_JOB_ID,${ALLOC_NODES[4]}:$SLURM_JOB_ID}"

export DOCKER_MODE="${DOCKER_MODE:-rootful}"
export PREPULL_R2EGYM_IMAGES="${PREPULL_R2EGYM_IMAGES:-true}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
export VLLM_SERVER_MODULE="${VLLM_SERVER_MODULE:-skyrl.backends.skyrl_train.inference_engines.vllm.vllm_server}"
export SKYRL_INFERENCE_ROUTER_PORT="${SKYRL_INFERENCE_ROUTER_PORT:-18080}"
export RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-r2egym-ta-mediumhard256-10epoch-nopf-${SLURM_JOB_ID}}"

SETUP_ENV="${SETUP_ENV:-sync}"
CLEANUP_ON_EXIT="${CLEANUP_ON_EXIT:-1}"

if [ "$SETUP_ENV" != "skip" ]; then
  bash "$SCRIPT_DIR/setup_env.sh" "$SETUP_ENV"
  source <(bash "$SCRIPT_DIR/setup_env.sh" print)
elif [ -z "${PYTHON_BIN:-}" ] || [ -z "${RAY_BIN:-}" ]; then
  source <(bash "$SCRIPT_DIR/setup_env.sh" print)
fi

cleanup() {
  if [ "$CLEANUP_ON_EXIT" = "1" ]; then
    bash "$WRAPPER" cleanup-stage all || true
  fi
}
trap cleanup EXIT

bash "$WRAPPER" cleanup-stage all
bash "$WRAPPER" prepare
bash "$WRAPPER" head
bash "$WRAPPER" ray
bash "$WRAPPER" rollout
bash "$WRAPPER" status
bash "$WRAPPER" driver
