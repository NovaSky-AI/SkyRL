#!/usr/bin/env bash
# Fleet shared run: Ray cluster setup (multi-node aware) + training launch
#
# Usage (from SkyPilot YAML run block):
#   bash skyrl-train/scripts/fleet-common-run.sh \
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
DATA_ROOT=""
CKPT_ROOT=""
USE_PYTHON_DIRECT=false
CUDA_ENV=""
SET_ULIMIT=false
NO_PYTORCH_ALLOC_CONF=false
NCCL_HEARTBEAT=""
ENTRYPOINT="integrations.fleet.entrypoints.main_fleet"
ENV_CLASS="fleet_task"
DATA_DIR_NAME=""
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
    --entrypoint) ENTRYPOINT="$2"; shift 2 ;;
    --env-class) ENV_CLASS="$2"; shift 2 ;;
    --data-dir-name) DATA_DIR_NAME="$2"; shift 2 ;;
    --) shift; HYDRA_OVERRIDES=("$@"); break ;;
    *) echo "ERROR: Unknown arg: $1"; exit 1 ;;
  esac
done

# Auto-detect data/ckpt root: /workspace if writable (RunPod), else $HOME (GCP, Lambda, etc.)
if [ -z "$DATA_ROOT" ]; then
  if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    DATA_ROOT="/workspace"
  else
    DATA_ROOT="$HOME"
  fi
fi
if [ -z "$CKPT_ROOT" ]; then
  CKPT_ROOT="$DATA_ROOT"
fi
DATA_DIR_NAME="${DATA_DIR_NAME:-$MODALITY}"

echo "=== Fleet Common Run ==="
echo "Entrypoint: $ENTRYPOINT"
echo "Env class: $ENV_CLASS"
echo "Data root: $DATA_ROOT"
echo "Data dir name: $DATA_DIR_NAME"
echo "Ckpt root: $CKPT_ROOT"

# Activate venv from repo root (upstream SkyRL layout)
source .venv/bin/activate

# --- Optional settings ---
if [ "$SET_ULIMIT" = true ]; then
  # Set open files limit. Try 1M first, fall back to hard limit if too high.
  ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
fi

# vLLM TP>1 uses pidfd_getfd for CUDA IPC weight sync between Ray workers.
# This requires ptrace access, which is blocked by default (ptrace_scope=1).
sudo sysctl -w kernel.yama.ptrace_scope=0 2>/dev/null || true

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
DATA_DIR="${DATA_ROOT}/data/fleet/${DATA_DIR_NAME}"

# --- System diagnostics ---
echo "=== System Diagnostics ==="
free -h
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv 2>/dev/null || true
echo "--- /dev/shm ---"
df -h /dev/shm 2>/dev/null || echo "/dev/shm not mounted"
ls -la /dev/shm/ 2>/dev/null | head -5 || true
echo "--- GPU Topology ---"
nvidia-smi topo -m 2>/dev/null || true
echo "--- cgroup memory limits ---"
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "No cgroup memory limit found"
echo "--- ulimits ---"
ulimit -a 2>/dev/null || true
echo "--- NCCL env vars ---"
env | grep -i NCCL || echo "No NCCL env vars set"
echo "--- kernel overcommit ---"
cat /proc/sys/vm/overcommit_memory 2>/dev/null || true
echo "=== End Diagnostics ==="

# --- wandb login ---
python3 -c "import wandb; wandb.login(relogin=True, key='$WANDB_API_KEY')"

# --- Fabric Manager check (NVSwitch GPUs: B200, H200 SXM) ---
# On non-GCP clouds (RunPod, Lambda, etc.), Fabric Manager is required for NVLink
# P2P on NVSwitch systems. Without it, dist.broadcast() in FSDP causes SIGKILL.
#
# On GCP, NVSwitch is managed at the HOST level — the guest VM does not have
# NVSwitch devices, so FM reports "NV_WARN_NOTHING_TO_DO" and cannot start.
# This is EXPECTED. NVLink P2P works through GCP's host-managed fabric without FM.
# GCP also provides a custom NCCL shim (gIB) that manages all NCCL configuration.
# Do NOT set NCCL_P2P_DISABLE or NCCL_NVLS_ENABLE on GCP with RDMA —
# the shim's "Guest Config Checker" expects these to be unset.
# NCCL_CUMEM_ENABLE=0 is set below for GCP WITHOUT RDMA to disable multicast.
ON_GCP=false
if [ -d "/usr/local/gib" ]; then
  ON_GCP=true
elif [ -f "/sys/class/dmi/id/product_name" ] && grep -qi "google" /sys/class/dmi/id/product_name 2>/dev/null; then
  ON_GCP=true
fi

FM_STATUS=$(systemctl is-active nvidia-fabricmanager 2>/dev/null || echo "unknown")
echo "Fabric Manager status: $FM_STATUS"
echo "On GCP: $ON_GCP"

if [ "$ON_GCP" = true ]; then
  echo "GCP detected — skipping Fabric Manager restart (host manages NVSwitch)"

  # GCP's deep learning images install /etc/profile.d/nccl_env.sh which auto-sources
  # /usr/local/gib/scripts/set_nccl_env.sh and adds /usr/local/gib/lib64 to LD_LIBRARY_PATH.
  # This sets NCCL_NET=gIB, forcing the gIB network plugin for RDMA/InfiniBand.
  #
  # Problem: gIB requires RDMA hardware (ConnectX NICs + multiple GPUDirect VPC networks).
  # SkyPilot provisions VMs with a single management NIC — no RDMA networking.
  # When NCCL_NET=gIB is forced but gIB can't init, NCCL fails with
  # "Failed to initialize any NET plugin" → SIGKILL during dist.broadcast().
  #
  # Fix: check for RDMA devices. If absent, strip gIB so NCCL falls back to
  # NVLink P2P for intra-node communication. Multi-node uses GKE with RDMA.
  if [ -d "/sys/class/infiniband" ] && [ "$(ls /sys/class/infiniband/ 2>/dev/null)" ]; then
    echo "RDMA devices found — keeping gIB for GPUDirect RDMA"
  else
    echo "No RDMA devices — disabling gIB"
    # Remove gIB from LD_LIBRARY_PATH (set by /etc/profile.d/nccl_env.sh)
    export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | sed 's|/usr/local/gib/lib64:||g; s|:/usr/local/gib/lib64||g; s|/usr/local/gib/lib64||g')
    # Unset NCCL_NET=gIB so NCCL can fall back to NVLink P2P
    unset NCCL_NET
    # Clear gIB-specific vars set by set_nccl_env.sh
    unset NCCL_CROSS_NIC NCCL_NET_GDR_LEVEL NCCL_P2P_NET_CHUNKSIZE NCCL_NVLS_CHUNKSIZE
    unset NCCL_IB_ADAPTIVE_ROUTING NCCL_IB_QPS_PER_CONNECTION NCCL_IB_TC NCCL_IB_FIFO_TC
    unset NCCL_TUNER_CONFIG_PATH
    # Disable CUDA multicast (requires NVSwitch fabric manager for GPU multicast
    # team setup). Without this, vLLM TP>1 hangs on CUDASymmetricMemory init.
    export NCCL_CUMEM_ENABLE=0
    echo "Cleared gIB NCCL env vars. Using NVLink P2P (intra-node)."
  fi
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "NCCL vars:"
  env | grep -i NCCL || echo "  (none)"

  # Ensure /dev/shm is large enough for NCCL IPC (some GCP images have small default)
  SHM_SIZE=$(df --output=size /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
  echo "Current /dev/shm size: ${SHM_SIZE}K"
  if [ -n "$SHM_SIZE" ] && [ "$SHM_SIZE" -lt 16777216 ]; then
    echo "WARNING: /dev/shm is only ${SHM_SIZE}K — remounting to 16G for NCCL"
    sudo mount -o remount,size=16G /dev/shm 2>&1 || echo "Failed to remount /dev/shm"
    df -h /dev/shm
  fi
elif [ "$FM_STATUS" != "active" ]; then
  echo "WARNING: Fabric Manager not active. Attempting restart..."
  sudo nvidia-smi -pm 1 2>&1 || true
  sudo systemctl stop nvidia-fabricmanager 2>&1 || true
  sleep 1
  sudo systemctl start nvidia-fabricmanager 2>&1 || true
  sleep 5
  FM_STATUS=$(systemctl is-active nvidia-fabricmanager 2>/dev/null || echo "unknown")
  echo "Fabric Manager status after restart: $FM_STATUS"
  if [ "$FM_STATUS" != "active" ]; then
    echo "=== WARNING: Fabric Manager failed to start ==="
    echo "Training may fail if this system has NVSwitch GPUs."
    sudo journalctl -u nvidia-fabricmanager --no-pager -n 10 2>&1 || true
  fi
fi

# --- Ray cluster setup (multi-node aware) ---
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_object_store_memory=10000000000
# Disable Ray's memory monitor to prevent spurious worker kills
export RAY_DISABLE_MEMORY_MONITOR=1
# NOTE: On GCP VMs without RDMA, gIB NCCL vars are stripped above.
# On GKE with RDMA, gIB is preserved for inter-node GPUDirect.

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
  # NUM_INFERENCE_ENGINES can be overridden via env var for TP>1 (engines = GPUs / TP)
  NUM_INFERENCE_ENGINES=${NUM_INFERENCE_ENGINES:-$TOTAL_GPUS}
  echo "=== Head node: $TOTAL_GPUS GPUs across ${SKYPILOT_NUM_NODES:-1} node(s), $NUM_INFERENCE_ENGINES inference engines ==="

  # Build training command
  CMD_ARGS=()
  if [ "$USE_PYTHON_DIRECT" = true ]; then
    CMD_ARGS=(python -m "$ENTRYPOINT")
  else
    CMD_ARGS=(uv run --isolated --extra "$INFERENCE_BACKEND" -m "$ENTRYPOINT")
  fi

  # Common hydra overrides (data paths, placement, strategy, checkpoints)
  CMD_ARGS+=(
    "data.train_data=['${DATA_DIR}/train.parquet']"
    "data.val_data=['${DATA_DIR}/validation.parquet']"
    "environment.env_class=$ENV_CLASS"
  )

  # fleet_task-specific: pass tasks_file path
  if [ "$ENV_CLASS" = "fleet_task" ]; then
    CMD_ARGS+=("environment.skyrl_gym.fleet_task.tasks_file=$TASKS_FILE")
  fi

  CMD_ARGS+=(
    trainer.placement.colocate_all=true
    trainer.strategy=fsdp2
    "trainer.placement.policy_num_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE"
    "trainer.placement.ref_num_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE"
    "trainer.placement.policy_num_nodes=${SKYPILOT_NUM_NODES:-1}"
    "trainer.placement.ref_num_nodes=${SKYPILOT_NUM_NODES:-1}"
    "generator.num_inference_engines=$NUM_INFERENCE_ENGINES"
    "trainer.ckpt_path=${CKPT_ROOT}/ckpts"
    "trainer.export_path=${CKPT_ROOT}/exports"
    trainer.dump_training_trajectories=true
  )

  # Append model-specific hydra overrides (passed after --)
  if [ ${#HYDRA_OVERRIDES[@]} -gt 0 ]; then
    CMD_ARGS+=("${HYDRA_OVERRIDES[@]}")
  fi

  export HYDRA_FULL_ERROR=1
  echo "=== Launching Training ==="
  set +e
  "${CMD_ARGS[@]}"
  EXIT_CODE=$?
  set -e

  if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Training failed (exit code $EXIT_CODE) ==="
    echo "--- dmesg (last 50 lines, unfiltered) ---"
    sudo dmesg -T 2>/dev/null | tail -50 || true
    echo "--- dmesg (OOM/kill/segfault) ---"
    sudo dmesg -T 2>/dev/null | grep -iE "oom|kill|out of memory|segfault|sigsegv|general protection|cgroup" | tail -20 || true
    echo "--- memory ---"
    free -h
    echo "--- GPU memory ---"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null || true
    echo "--- /dev/shm after crash ---"
    df -h /dev/shm 2>/dev/null || true
    echo "--- cgroup memory events ---"
    cat /sys/fs/cgroup/memory.events 2>/dev/null || cat /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null || true
    echo "--- Ray worker logs (last errors) ---"
    grep -r "SIGKILL\|SIGABRT\|SIGSEGV\|SYSTEM_ERROR\|RuntimeError\|NCCL" /tmp/ray/session_latest/logs/ 2>/dev/null | tail -30 || true
    exit $EXIT_CODE
  fi

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
