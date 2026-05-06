#!/usr/bin/env bash
set -euo pipefail

# Start the external vLLM rollout servers used by the ThunderAgent 32B recipe.
# This script intentionally lives beside the ThunderAgent recipe so the recipe
# does not depend on the old runtime-full worktree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
TMP_LOG_ROOT="${TMP_LOG_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/tmp_logs}"

resolve_python_bin() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    printf '%s\n' "$REPO_ROOT/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  command -v python
}

trim_csv_array() {
  local -n ref="$1"
  local idx=""
  for idx in "${!ref[@]}"; do
    ref[$idx]="$(printf '%s' "${ref[$idx]}" | xargs)"
  done
}

resolve_writable_runtime_root() {
  local preferred="${1:-}"
  local candidate=""
  for candidate in \
    "$preferred" \
    "/tmp/$USER/skyrl_runtime" \
    "$(cd "$REPO_ROOT/.." && pwd)/tmp_runtime/$USER"; do
    [ -n "$candidate" ] || continue
    if mkdir -p "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "Failed to find a writable runtime root" >&2
  exit 1
}

ensure_soft_nofile_limit() {
  local requested_soft="$1"
  [ -n "$requested_soft" ] || return 0
  if ! [[ "$requested_soft" =~ ^[0-9]+$ ]]; then
    echo "Invalid ROLLOUT_NOFILE_SOFT: $requested_soft" >&2
    exit 1
  fi
  current_soft="$(ulimit -Sn)"
  if [ "$current_soft" != "unlimited" ] && [ "$current_soft" -lt "$requested_soft" ]; then
    ulimit -Sn "$requested_soft"
  fi
}

server_source_name() {
  local idx="$1"
  local suffixes=(a b c d e f g h i j k l m n o p)
  if [ "$idx" -lt "${#suffixes[@]}" ]; then
    printf 'rollout_%s' "${suffixes[$idx]}"
    return
  fi
  printf 'rollout_%02d' "$idx"
}

discover_visible_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    local -a visible=()
    IFS=',' read -r -a visible <<<"$CUDA_VISIBLE_DEVICES"
    trim_csv_array visible
    printf '%s\n' "${visible[@]}"
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader | awk '{$1=$1; print}'
}

build_default_gpu_groups() {
  local server_count="$1"
  local tp_size="$2"
  local -a visible_gpus=()
  local required_gpus=""
  local cursor=0
  local server_idx=""
  local group_idx=""
  local -a group=()

  mapfile -t visible_gpus < <(discover_visible_gpus)
  required_gpus=$((server_count * tp_size))
  if [ "${#visible_gpus[@]}" -lt "$required_gpus" ]; then
    echo "Need at least $required_gpus visible GPUs for $server_count servers with TP=$tp_size, found ${#visible_gpus[@]}" >&2
    return 1
  fi
  for ((server_idx = 0; server_idx < server_count; server_idx++)); do
    group=()
    for ((group_idx = 0; group_idx < tp_size; group_idx++)); do
      group+=("${visible_gpus[$cursor]}")
      cursor=$((cursor + 1))
    done
    printf '%s\n' "$(IFS=,; echo "${group[*]}")"
  done
}

cleanup_existing() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"$port" || true)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
  fi
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
    sleep 2
  fi
}

wait_for_health() {
  local url="$1"
  local name="$2"
  local pid="$3"
  local timeout_sec="${ROLLOUT_HEALTH_TIMEOUT_SEC:-900}"
  local waited=0
  while [ "$waited" -lt "$timeout_sec" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name process exited before becoming healthy" >&2
      return 1
    fi
    if curl -sf "$url/health" >/dev/null; then
      echo "$name healthy at $url"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "Timed out waiting for $name at $url" >&2
  return 1
}

STORAGE_ROOT="${STORAGE_ROOT:-$HOME}"
MODEL_ROOT="${MODEL_ROOT:-$STORAGE_ROOT/models}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-32B}"
ROLLOUT_HOST="${ROLLOUT_HOST:-0.0.0.0}"
ROLLOUT_SERVER_PORTS_CSV="${ROLLOUT_SERVER_PORTS_CSV:-18000,18001,18002,18003}"
ROLLOUT_GPU_GROUPS_SPEC="${ROLLOUT_GPU_GROUPS_SPEC:-}"
TP_SIZE="${TP_SIZE:-${ROLLOUT_TP_SIZE:-2}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-$REPO_ROOT/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2}"
RUN_NAME="${RUN_NAME:-r2egym-ta-rollout}"
LOG_DIR="${LOG_DIR:-$TMP_LOG_ROOT/$RUN_NAME/rollout}"
SCRATCH_ROOT="$(resolve_writable_runtime_root "${SCRATCH_ROOT:-/scratch/$USER/skyrl_runtime/${RUN_NAME}-rollout}")"
PYTHON_BIN="$(resolve_python_bin)"
ROLLOUT_NOFILE_SOFT="${ROLLOUT_NOFILE_SOFT:-131072}"
VLLM_SERVER_MODULE="${VLLM_SERVER_MODULE:-skyrl.backends.skyrl_train.inference_engines.vllm.vllm_server}"

mkdir -p "$LOG_DIR" "$SCRATCH_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_ROOT/uv}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRATCH_ROOT/torchinductor}"
export TRITON_HOME="${TRITON_HOME:-$SCRATCH_ROOT/triton-home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_ROOT/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/xdg-cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$SCRATCH_ROOT/xdg-config}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$SCRATCH_ROOT/vllm-cache}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$SCRATCH_ROOT/vllm-config}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export VLLM_USE_STANDALONE_COMPILE="${VLLM_USE_STANDALONE_COMPILE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-1}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HOME="${HF_HOME:-$STORAGE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

ensure_soft_nofile_limit "$ROLLOUT_NOFILE_SOFT"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "Model path is missing config.json: $MODEL_PATH" >&2
  exit 1
fi

detect_socket_ifname() {
  local target_ip="${1:-}"
  local ifname=""
  if [ -n "$target_ip" ] && command -v ip >/dev/null 2>&1; then
    ifname="$(ip route get "$target_ip" 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i=="dev" && i<NF) {print $(i+1); exit}}')"
  fi
  if [ -z "$ifname" ] && command -v ip >/dev/null 2>&1; then
    ifname="$(ip route show default 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i=="dev" && i<NF) {print $(i+1); exit}}')"
  fi
  if [ -n "$ifname" ]; then
    printf '%s\n' "$ifname"
    return
  fi
  ls /sys/class/net | grep -Ev '^(lo|docker|br-|veth)' | head -n1
}

SOCKET_IFNAME_TARGET_IP="${SOCKET_IFNAME_TARGET_IP:-${RAY_HEAD_IP:-}}"
DEFAULT_SOCKET_IFNAME="$(detect_socket_ifname "$SOCKET_IFNAME_TARGET_IP")"
if [ "${SKYRL_RESPECT_SOCKET_IFNAME:-0}" != "1" ]; then
  export NCCL_SOCKET_IFNAME="$DEFAULT_SOCKET_IFNAME"
  export GLOO_SOCKET_IFNAME="$DEFAULT_SOCKET_IFNAME"
else
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$DEFAULT_SOCKET_IFNAME}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$DEFAULT_SOCKET_IFNAME}"
fi
echo "Using socket interfaces: NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME target=$SOCKET_IFNAME_TARGET_IP"

IFS=',' read -r -a SERVER_PORTS <<<"$ROLLOUT_SERVER_PORTS_CSV"
trim_csv_array SERVER_PORTS
if [ -n "$ROLLOUT_GPU_GROUPS_SPEC" ]; then
  IFS=';' read -r -a SERVER_GPU_GROUPS <<<"$ROLLOUT_GPU_GROUPS_SPEC"
  trim_csv_array SERVER_GPU_GROUPS
else
  mapfile -t SERVER_GPU_GROUPS < <(build_default_gpu_groups "${#SERVER_PORTS[@]}" "$TP_SIZE")
fi

if [ "${#SERVER_GPU_GROUPS[@]}" -ne "${#SERVER_PORTS[@]}" ]; then
  echo "ROLLOUT_GPU_GROUPS_SPEC count (${#SERVER_GPU_GROUPS[@]}) does not match port count (${#SERVER_PORTS[@]})" >&2
  exit 1
fi

SERVER_PIDS=()
cleanup() {
  if [ "${#SERVER_PIDS[@]}" -gt 0 ]; then
    kill "${SERVER_PIDS[@]}" 2>/dev/null || true
    wait "${SERVER_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for idx in "${!SERVER_PORTS[@]}"; do
  cleanup_existing "${SERVER_PORTS[$idx]}"
done

for idx in "${!SERVER_PORTS[@]}"; do
  source_name="$(server_source_name "$idx")"
  log_file="$LOG_DIR/${source_name}.log"
  extra_vllm_args=()
  if [ "$ROLLOUT_ENFORCE_EAGER" = true ]; then
    extra_vllm_args+=(--enforce-eager)
  fi
  : >"$log_file"
  echo "Starting $source_name port=${SERVER_PORTS[$idx]} gpus=${SERVER_GPU_GROUPS[$idx]} log=$log_file"
  SKYRL_EXTERNAL_SERVER_IDX="$idx" CUDA_VISIBLE_DEVICES="${SERVER_GPU_GROUPS[$idx]}" "$PYTHON_BIN" \
    -m "$VLLM_SERVER_MODULE" \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --host "$ROLLOUT_HOST" \
    --port "${SERVER_PORTS[$idx]}" \
    --seed 42 \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-sleep-mode \
    --max-num_batched_tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --trust-remote-code \
    --chat-template "$CHAT_TEMPLATE_PATH" \
    --distributed-executor-backend mp \
    --worker-extension-cls skyrl.backends.skyrl_train.inference_servers.vllm_worker.WorkerWrap \
    "${extra_vllm_args[@]}" \
    >"$log_file" 2>&1 &
  SERVER_PIDS+=("$!")
done

for idx in "${!SERVER_PORTS[@]}"; do
  source_name="$(server_source_name "$idx")"
  wait_for_health "http://127.0.0.1:${SERVER_PORTS[$idx]}" "$source_name" "${SERVER_PIDS[$idx]}"
done

echo "External rollout servers ready:"
for idx in "${!SERVER_PORTS[@]}"; do
  source_name="$(server_source_name "$idx")"
  echo "  $source_name: http://$(hostname -f):${SERVER_PORTS[$idx]} (gpus=${SERVER_GPU_GROUPS[$idx]})"
done
echo "Logs: $LOG_DIR"

wait "${SERVER_PIDS[@]}"
