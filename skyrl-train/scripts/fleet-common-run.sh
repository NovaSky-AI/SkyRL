#!/usr/bin/env bash
# Fleet shared run: Ray cluster setup (multi-node aware) + training launch
#
# Usage (from SkyPilot YAML run block):
#   bash skyrl-train/scripts/fleet-common-run.sh \
#     --data-root /workspace --ckpt-root /workspace \
#     --use-python-direct --cuda-env "$HOME/.cuda_env" \
#     --set-ulimit --no-pytorch-alloc-conf -- \
#     trainer.policy.model.path="Qwen/Qwen3.5-9B" \
#     trainer.epochs=20 ...
#
# Multi-node:
#   Rank 0 (head): starts Ray head, launches training
#   Rank >0 (workers): joins Ray cluster, sleeps
#
# Required env vars: WANDB_API_KEY, MODALITY, INFERENCE_BACKEND,
#   SKYPILOT_NUM_GPUS_PER_NODE, SKYPILOT_NODE_IPS
# Optional env vars: SKYPILOT_NUM_NODES, SKYPILOT_NODE_RANK
set -euo pipefail

# Defaults
DATA_ROOT="$HOME"
CKPT_ROOT="$HOME"
USE_PYTHON_DIRECT=false
CUDA_ENV=""
SET_ULIMIT=false
NO_PYTORCH_ALLOC_CONF=false
NCCL_HEARTBEAT=""
HYDRA_OVERRIDES=()

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
    --use-python-direct) USE_PYTHON_DIRECT=true; shift ;;
    --cuda-env) CUDA_ENV="$2"; shift 2 ;;
    --set-ulimit) SET_ULIMIT=true; shift ;;
    --no-pytorch-alloc-conf) NO_PYTORCH_ALLOC_CONF=true; shift ;;
    --nccl-heartbeat) NCCL_HEARTBEAT="$2"; shift 2 ;;
    --) shift; HYDRA_OVERRIDES=("$@"); break ;;
    *) echo "ERROR: Unknown arg: $1"; exit 1 ;;
  esac
done

cd skyrl-train
source .venv/bin/activate

# --- Optional settings ---
if [ "$SET_ULIMIT" = true ]; then
  ulimit -n 65536
fi

if [ -n "$CUDA_ENV" ]; then
  source "$CUDA_ENV" 2>/dev/null || true
fi

if [ "$NO_PYTORCH_ALLOC_CONF" = false ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

if [ -n "$NCCL_HEARTBEAT" ]; then
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="$NCCL_HEARTBEAT"
fi

TMP_DIR="${CKPT_ROOT}/skyrl-tmp"
mkdir -p "$TMP_DIR"
export TMPDIR="$TMP_DIR"

TASKS_FILE="${DATA_ROOT}/data/fleet/tasks_${MODALITY}.json"
DATA_DIR="${DATA_ROOT}/data/fleet/${MODALITY}"

# --- wandb login ---
python3 -c "import wandb; wandb.login(relogin=True, key='$WANDB_API_KEY')"

# --- Ray cluster setup (multi-node aware) ---
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_object_store_memory=10000000000

read -r head_ip _ <<< "$SKYPILOT_NODE_IPS"

wait_for_ray() {
  local address=$1
  for _ in $(seq 1 24); do
    if ray status --address "$address" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: Ray cluster at $address failed to become ready" >&2
  return 1
}

if [ "${SKYPILOT_NODE_RANK:-0}" = "0" ]; then
  # === Head node: start Ray head + launch training ===
  if ! ray status --address 127.0.0.1:6479 >/dev/null 2>&1; then
    ray start --head --disable-usage-stats --port 6479 --object-store-memory=10000000000
  fi
  wait_for_ray 127.0.0.1:6479

  TOTAL_GPUS=$((SKYPILOT_NUM_GPUS_PER_NODE * ${SKYPILOT_NUM_NODES:-1}))
  export TOTAL_GPUS
  echo "=== Head node: $TOTAL_GPUS GPUs across ${SKYPILOT_NUM_NODES:-1} node(s) ==="

  # Build training command
  CMD_ARGS=()
  if [ "$USE_PYTHON_DIRECT" = true ]; then
    CMD_ARGS=(python -m integrations.fleet.entrypoints.main_fleet)
  else
    CMD_ARGS=(uv run --isolated --extra "$INFERENCE_BACKEND" -m integrations.fleet.entrypoints.main_fleet)
  fi

  # Common hydra overrides (data paths, placement, strategy)
  CMD_ARGS+=(
    "data.train_data=['${DATA_DIR}/train.parquet']"
    "data.val_data=['${DATA_DIR}/validation.parquet']"
    environment.env_class=fleet_task
    "environment.skyrl_gym.fleet_task.tasks_file=$TASKS_FILE"
    trainer.placement.colocate_all=true
    trainer.strategy=fsdp2
    "trainer.placement.policy_num_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE"
    "trainer.placement.ref_num_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE"
    "trainer.placement.policy_num_nodes=${SKYPILOT_NUM_NODES:-1}"
    "trainer.placement.ref_num_nodes=${SKYPILOT_NUM_NODES:-1}"
    "generator.num_inference_engines=$TOTAL_GPUS"
  )

  # Append model-specific hydra overrides (passed after --)
  if [ ${#HYDRA_OVERRIDES[@]} -gt 0 ]; then
    CMD_ARGS+=("${HYDRA_OVERRIDES[@]}")
  fi

  echo "=== Launching Training ==="
  exec "${CMD_ARGS[@]}"

else
  # === Worker node: join Ray cluster and wait ===
  echo "=== Worker node (rank ${SKYPILOT_NODE_RANK}), joining Ray cluster at $head_ip:6479 ==="
  if ! ray status --address "$head_ip:6479" >/dev/null 2>&1; then
    ray start --address "$head_ip:6479" --disable-usage-stats
  fi
  wait_for_ray "$head_ip:6479"
  echo "Worker node joined. Sleeping..."
  sleep infinity
fi
