#!/usr/bin/env bash
set -euo pipefail

# Cross-job launcher for the R2EGym Qwen3-32B ThunderAgent recipe.
#
# Canonical path:
#   cleanup-stage all -> prepare -> head -> ray -> rollout -> status -> driver
#
# This script replaces the old runtime-full handoff wrapper for this run.
# The training driver calls the script-local run_trainer.sh helper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

ACTION="${1:-status}"
ACTION_ARG="${2:-}"

if [ -z "${RECIPE_HOME:-}" ]; then
  if [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
    RECIPE_HOME="/scratch/$USER/skyrl-thunder-agent"
  else
    RECIPE_HOME="$HOME/.cache/skyrl-thunder-agent"
  fi
fi

DEFAULT_RECIPE_VENV="${RECIPE_VENV:-$RECIPE_HOME/venvs/skyrl-ta-pr-core-vllm0201-cu129}"
if [ -x "$DEFAULT_RECIPE_VENV/bin/python" ]; then
  DEFAULT_PYTHON_BIN="$DEFAULT_RECIPE_VENV/bin/python"
else
  DEFAULT_PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
RAY_BIN="${RAY_BIN:-$(dirname "$PYTHON_BIN")/ray}"
PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"

MERGED_JOB_ID="${MERGED_JOB_ID:-${JOB_ID:-${SLURM_JOB_ID:-}}}"
MERGED_NODE="${MERGED_NODE:-}"
ROLLOUT_JOB_ID="${ROLLOUT_JOB_ID:-}"
ROLLOUT_NODE="${ROLLOUT_NODE:-}"
TRAINER_NODE_SPECS="${TRAINER_NODE_SPECS:-}"

RAY_PORT="${RAY_PORT:-6381}"
HEAD_CPUS="${HEAD_CPUS:-8}"
HEAD_GPUS="${HEAD_GPUS:-0}"
TRAINER_CPUS="${TRAINER_CPUS:-176}"
TRAINER_GPUS="${TRAINER_GPUS:-8}"
ROLLOUT_CPUS="${ROLLOUT_CPUS:-100}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
ROLLOUT_SERVER_PORTS_CSV="${ROLLOUT_SERVER_PORTS_CSV:-18000,18001,18002,18003}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_ENGINES="${ROLLOUT_ENGINES:-4}"
ROLLOUT_GPU_GROUPS_SPEC="${ROLLOUT_GPU_GROUPS_SPEC:-}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
VLLM_SERVER_MODULE="${VLLM_SERVER_MODULE:-skyrl.backends.skyrl_train.inference_engines.vllm.vllm_server}"
SKYRL_INFERENCE_ROUTER_PORT="${SKYRL_INFERENCE_ROUTER_PORT:-18080}"

DOCKER_MODE="${DOCKER_MODE:-rootful}"
HARBOR_DOCKER_SHARED_NETWORK_NAME="${HARBOR_DOCKER_SHARED_NETWORK_NAME:-harbor-r2egym-shared}"
HARBOR_DOCKER_SHARED_NETWORK_SUBNET="${HARBOR_DOCKER_SHARED_NETWORK_SUBNET:-172.30.0.0/16}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRATCH:-/scratch/$USER}}"
HARBOR_SHARED_UV_CACHE_HOST_DIR="${HARBOR_SHARED_UV_CACHE_HOST_DIR:-$RUNTIME_ROOT/harbor-uv-cache}"
HARBOR_SHARED_UV_CACHE_ENV_DIR="${HARBOR_SHARED_UV_CACHE_ENV_DIR:-/harbor-shared/uv-cache}"
HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME="${HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME:-$RUNTIME_ROOT/harbor-mini-swe-home}"
HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME="${HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME:-/tmp/harbor-mini-swe-home}"
HARBOR_SHARED_UV_PYTHON_HOST_DIR="${HARBOR_SHARED_UV_PYTHON_HOST_DIR:-/home/$USER/.local/share/uv/python}"
HARBOR_SHARED_UV_PYTHON_ENV_DIR="${HARBOR_SHARED_UV_PYTHON_ENV_DIR:-$HARBOR_SHARED_UV_PYTHON_HOST_DIR}"
HARBOR_MINI_SWE_AGENT_PACKAGE="${HARBOR_MINI_SWE_AGENT_PACKAGE:-git+https://github.com/li-boxuan/mini-swe-agent.git@8e8a515fdcecf3a8e45c3909f7f196bfe18ca89a}"
HARBOR_MINI_SWE_AGENT_UV_OFFLINE="${HARBOR_MINI_SWE_AGENT_UV_OFFLINE:-0}"
PREPULL_R2EGYM_IMAGES="${PREPULL_R2EGYM_IMAGES:-true}"
PREPULL_R2EGYM_IMAGE_RETRIES="${PREPULL_R2EGYM_IMAGE_RETRIES:-3}"
PREPULL_R2EGYM_IMAGE_RETRY_SLEEP="${PREPULL_R2EGYM_IMAGE_RETRY_SLEEP:-30}"

DATA_ROOT="${DATA_ROOT:-$HOME/data/harbor}"
TRAIN_DATA="${TRAIN_DATA:-['$DATA_ROOT/r2egym-train256-medium-hard-v1']}"
EVAL_DATA="${EVAL_DATA:-['$DATA_ROOT/r2egym-eval64-medium-hard-v1']}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-32B}"
MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-256}"
MAX_EVAL_TASKS="${MAX_EVAL_TASKS:-64}"
HARBOR_AGENT_MAX_TURNS="${HARBOR_AGENT_MAX_TURNS:-25}"
HARBOR_AGENT_TEMPERATURE="${HARBOR_AGENT_TEMPERATURE:-0.3}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-9000}"
MINI_SWE_MODEL_TIMEOUT_SEC="${MINI_SWE_MODEL_TIMEOUT_SEC:-1200}"
HARBOR_VERIFIER_TIMEOUT_MAX_ATTEMPTS="${HARBOR_VERIFIER_TIMEOUT_MAX_ATTEMPTS:-1}"
HARBOR_HARD_FAILURE_EXCEPTION_TYPES="${HARBOR_HARD_FAILURE_EXCEPTION_TYPES:-RewardFileNotFoundError,VerifierTimeoutError}"
HARBOR_TASK_CIRCUIT_BREAKER_ENABLED="${HARBOR_TASK_CIRCUIT_BREAKER_ENABLED:-1}"
HARBOR_TASK_CIRCUIT_BREAKER_THRESHOLD="${HARBOR_TASK_CIRCUIT_BREAKER_THRESHOLD:-2}"
FULL_EPOCHS="${FULL_EPOCHS:-10}"
EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-4}"
USE_KL_LOSS="${USE_KL_LOSS:-false}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
CKPT_ROOT_OVERRIDE="${CKPT_ROOT_OVERRIDE:-$RUNTIME_ROOT/harbor_ckpts}"
CKPT_INTERVAL="${CKPT_INTERVAL:-4}"
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:--1}"
EXPORT_ROOT_OVERRIDE="${EXPORT_ROOT_OVERRIDE:-}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-none}"
TRAINER_RESUME_PATH="${TRAINER_RESUME_PATH:-}"
RUN_PREFLIGHT_CHECKS="${RUN_PREFLIGHT_CHECKS:-false}"
AGENT_RUNTIME_PREFLIGHT="${AGENT_RUNTIME_PREFLIGHT:-false}"
RUN_HARBOR_DOCKER_CONCURRENCY_SMOKE_PREPARE="${RUN_HARBOR_DOCKER_CONCURRENCY_SMOKE_PREPARE:-false}"
THUNDERAGENT_WATCHDOG_ENABLED="${THUNDERAGENT_WATCHDOG_ENABLED:-0}"
SKYRL_WORKER_NCCL_TIMEOUT_IN_S="${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-3600}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME_OVERRIDE:-r2egym-ta-mediumhard256-10epoch-nopf-${RUN_TS}}"
if [ -z "${RUN_SHORT_ID:-}" ]; then
  RUN_SHORT_ID="r2e-$(printf '%s' "$RUN_NAME" | sha1sum | cut -c1-10)"
fi
LOG_DIR="${LOG_DIR_OVERRIDE:-$WORKSPACE_ROOT/tmp_logs/$RUN_NAME}"
RUN_ARTIFACT_ROOT="${RUN_ARTIFACT_ROOT:-$RUNTIME_ROOT/harbor_run_artifacts}"
RUN_ARTIFACT_DIR="$RUN_ARTIFACT_ROOT/$RUN_NAME"
STATE_DIR="$LOG_DIR/state"
LOCAL_TMP_DIR="$LOG_DIR/local-tmp"
RAY_LOG="$LOG_DIR/launcher_ray.log"
ROLLOUT_LOG="$LOG_DIR/launcher_rollout.log"
TRAIN_DRIVER_LOG="$LOG_DIR/launcher_train_driver.log"
HEAD_LOG="$LOG_DIR/launcher_head.log"
PREPARE_LOG="$LOG_DIR/launcher_prepare.log"
ROLLOUT_LOG_DIR="$LOG_DIR/rollout"
HEAD_RAY_TMP_DIR="${HEAD_RAY_TMP_DIR:-/tmp/rh-$RUN_SHORT_ID}"
TRAINER_RAY_TMP_DIR_ROOT="${TRAINER_RAY_TMP_DIR_ROOT:-/tmp/rw-$RUN_SHORT_ID}"
MERGED_TMP_DIR="${MERGED_TMP_DIR:-$RUNTIME_ROOT/${RUN_SHORT_ID}-tmp}"
TRAIN_RUNTIME_SCRATCH_ROOT="${TRAIN_RUNTIME_SCRATCH_ROOT:-$RUNTIME_ROOT/skyrl_runtime/${RUN_SHORT_ID}-train}"
ROLLOUT_RUNTIME_SCRATCH_ROOT="${ROLLOUT_RUNTIME_SCRATCH_ROOT:-$RUNTIME_ROOT/skyrl_runtime/${RUN_SHORT_ID}-rollout}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TRAIN_RUNTIME_SCRATCH_ROOT/triton-cache}"
DOCKER_HOST="${WRAPPER_DOCKER_HOST:-unix:///var/run/docker.sock}"

TRAINER_NODES=()
TRAINER_JOB_IDS=()
TRAINER_IPS=()
ROLLOUT_NODE_EFFECTIVE=""
ROLLOUT_JOB_ID_EFFECTIVE=""
MERGED_IP=""
ROLLOUT_IP=""
SOCKET_IFNAME_PEER_IP=""
ROLLOUT_PORTS=()

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

trim_csv_array() {
  local -n ref="$1"
  local idx=""
  for idx in "${!ref[@]}"; do
    ref[$idx]="$(printf '%s' "${ref[$idx]}" | xargs)"
  done
}

dataset_spec_paths() {
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY' "$1"
import ast
import os
import sys

raw = sys.argv[1]
parsed = ast.literal_eval(raw)
if isinstance(parsed, str):
    items = [parsed]
elif isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
    items = parsed
else:
    raise SystemExit(f"Dataset spec must be a string or list of strings, got {raw!r}")
for item in items:
    print(os.path.expanduser(item))
PY
}

parse_trainer_specs() {
  local spec=""
  local node=""
  local job_id=""
  [ -n "$TRAINER_NODE_SPECS" ] || {
    echo "TRAINER_NODE_SPECS is required. Format: node:jobid,node:jobid,node:jobid,node:jobid" >&2
    exit 1
  }
  IFS=',' read -r -a raw_specs <<<"$TRAINER_NODE_SPECS"
  trim_csv_array raw_specs
  for spec in "${raw_specs[@]}"; do
    [ -n "$spec" ] || continue
    node="${spec%%:*}"
    job_id="${spec##*:}"
    if [ -z "$node" ] || [ -z "$job_id" ] || [ "$node" = "$job_id" ]; then
      echo "Invalid trainer spec: $spec" >&2
      exit 1
    fi
    TRAINER_NODES+=("$node")
    TRAINER_JOB_IDS+=("$job_id")
  done
  if [ "${#TRAINER_NODES[@]}" -ne 4 ]; then
    echo "Expected exactly 4 trainer specs, got ${#TRAINER_NODES[@]}: ${TRAINER_NODE_SPECS}" >&2
    exit 1
  fi
}

node_ip() {
  local job_id="$1"
  local node="$2"
  srun --jobid "$job_id" --overlap --overcommit --immediate=10 -w "$node" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
    bash -lc "hostname -I | tr ' ' '\n' | grep '^172\.27\.' | head -n1 || hostname -I | awk '{print \$1}'"
}

stage_pid_file() {
  printf '%s/%s.client.pid\n' "$STATE_DIR" "$1"
}

stage_pid_is_running() {
  local pid_file=""
  local pid=""
  pid_file="$(stage_pid_file "$1")"
  [ -f "$pid_file" ] || return 1
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_detached_client() {
  local stage="$1"
  local log_file="$2"
  shift 2
  local pid_file=""
  local pid=""
  mkdir -p "$STATE_DIR" "$(dirname "$log_file")"
  pid_file="$(stage_pid_file "$stage")"
  if stage_pid_is_running "$stage"; then
    echo "Stage is already running: $stage (pid $(cat "$pid_file"))" >&2
    exit 1
  fi
  if [ ! -f "$log_file" ]; then
    : >"$log_file"
  fi
  setsid "$@" </dev/null >>"$log_file" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$pid_file"
  echo "Started $stage client pid=$pid log=$log_file"
}

wait_for_http() {
  local url="$1"
  local name="$2"
  local timeout="${3:-900}"
  local waited=0
  while [ "$waited" -lt "$timeout" ]; do
    if curl -sf "$url" >/dev/null; then
      echo "$name ready: $url"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "Timed out waiting for $name: $url" >&2
  return 1
}

ensure_launch_requirements() {
  require_cmd srun
  require_cmd curl
  require_env MERGED_JOB_ID
  require_env MERGED_NODE
  if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python env not found: $PYTHON_BIN" >&2
    exit 1
  fi
  if [ ! -x "$RAY_BIN" ]; then
    echo "Ray CLI not found: $RAY_BIN" >&2
    exit 1
  fi
  parse_trainer_specs
  ROLLOUT_NODE_EFFECTIVE="${ROLLOUT_NODE:-$MERGED_NODE}"
  ROLLOUT_JOB_ID_EFFECTIVE="${ROLLOUT_JOB_ID:-$MERGED_JOB_ID}"
  mkdir -p "$LOG_DIR" "$STATE_DIR" "$LOCAL_TMP_DIR" "$RUN_ARTIFACT_DIR" "$ROLLOUT_LOG_DIR"
  MERGED_IP="$(node_ip "$MERGED_JOB_ID" "$MERGED_NODE")"
  local idx=""
  for idx in "${!TRAINER_NODES[@]}"; do
    TRAINER_IPS+=("$(node_ip "${TRAINER_JOB_IDS[$idx]}" "${TRAINER_NODES[$idx]}")")
  done
  SOCKET_IFNAME_PEER_IP="${TRAINER_IPS[0]}"
  ROLLOUT_IP="$MERGED_IP"
  if [ "$ROLLOUT_NODE_EFFECTIVE" != "$MERGED_NODE" ] || [ "$ROLLOUT_JOB_ID_EFFECTIVE" != "$MERGED_JOB_ID" ]; then
    ROLLOUT_IP="$(node_ip "$ROLLOUT_JOB_ID_EFFECTIVE" "$ROLLOUT_NODE_EFFECTIVE")"
  fi
  IFS=',' read -r -a ROLLOUT_PORTS <<<"$ROLLOUT_SERVER_PORTS_CSV"
  trim_csv_array ROLLOUT_PORTS
}

print_banner() {
  cat <<EOF
ThunderAgent R2EGym 32B recipe
  action:        $ACTION${ACTION_ARG:+ $ACTION_ARG}
  repo_root:     $REPO_ROOT
  python:        $PYTHON_BIN
  merged:        $MERGED_NODE job=$MERGED_JOB_ID ip=$MERGED_IP
  rollout:       $ROLLOUT_NODE_EFFECTIVE job=$ROLLOUT_JOB_ID_EFFECTIVE ip=$ROLLOUT_IP
  trainers:      $TRAINER_NODE_SPECS
  socket_peer:   $SOCKET_IFNAME_PEER_IP
  run_name:      $RUN_NAME
  log_dir:       $LOG_DIR
  model_path:    $MODEL_PATH
  train_data:    $TRAIN_DATA
  eval_data:     $EVAL_DATA
  docker_mode:   $DOCKER_MODE
  docker_host:   $DOCKER_HOST
  prepull_images: $PREPULL_R2EGYM_IMAGES
  vllm_server:   $VLLM_SERVER_MODULE
  eager:         $ROLLOUT_ENFORCE_EAGER
EOF
}

run_prepare() {
  {
    echo "Checking Python packages"
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import importlib
for name in ["ray", "torch", "vllm", "fastapi", "uvicorn"]:
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'ok')}")
from examples.train.thunder_agent.main_harbor_thunder_agent import HarborThunderAgentFullyAsyncExp
from examples.train.thunder_agent.skyrl_integration.generator import ThunderAgentHarborGenerator
from skyrl.backends.skyrl_train.inference_servers.vllm_worker import WorkerWrap
print("pr_core_imports=ok")
PY
    if [ ! -f "$MODEL_PATH/config.json" ]; then
      echo "Missing model config: $MODEL_PATH/config.json" >&2
      exit 1
    fi
    while IFS= read -r path; do
      if [ ! -d "$path" ]; then
        echo "Missing dataset directory: $path" >&2
        exit 1
      fi
    done < <(dataset_spec_paths "$TRAIN_DATA")
    while IFS= read -r path; do
      if [ ! -d "$path" ]; then
        echo "Missing eval dataset directory: $path" >&2
        exit 1
      fi
    done < <(dataset_spec_paths "$EVAL_DATA")
    echo "prepare OK"
  } | tee "$PREPARE_LOG"
}

run_head() {
  local train_data_escaped=""
  local eval_data_escaped=""
  printf -v train_data_escaped '%q' "$TRAIN_DATA"
  printf -v eval_data_escaped '%q' "$EVAL_DATA"
  start_detached_client "head" "$HEAD_LOG" \
    srun --jobid "$MERGED_JOB_ID" --overlap --overcommit --immediate=10 -w "$MERGED_NODE" --ntasks=1 --nodes=1 --cpus-per-task=4 --gres=gpu:0 \
    bash -lc "set -euo pipefail
      export PATH='$PYTHON_BIN_DIR':\$HOME/.local/bin:\$PATH
      export DOCKER_MODE='$DOCKER_MODE'
      export DOCKER_HOST='$DOCKER_HOST'
      mkdir -p '$HARBOR_SHARED_UV_CACHE_HOST_DIR' '$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME' '$MERGED_TMP_DIR'
      if [ '$DOCKER_MODE' != rootful ]; then
        echo 'This recipe expects DOCKER_MODE=rootful.' >&2
        exit 1
      fi
      if ! docker info >/dev/null 2>&1; then
        if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1 && command -v setfacl >/dev/null 2>&1; then
          sudo -n setfacl -m u:'$USER':rw /var/run/docker.sock /run/docker.sock 2>/dev/null || true
        fi
      fi
      if ! docker info >/dev/null; then
        echo 'Docker is not usable. For rootful mode, ensure the current Slurm job can access /var/run/docker.sock.' >&2
        ls -l /var/run/docker.sock /run/docker.sock >&2 || true
        id >&2
        exit 1
      fi
      docker network inspect '$HARBOR_DOCKER_SHARED_NETWORK_NAME' >/dev/null 2>&1 || \
        docker network create --subnet '$HARBOR_DOCKER_SHARED_NETWORK_SUBNET' '$HARBOR_DOCKER_SHARED_NETWORK_NAME'
      if [ '$PREPULL_R2EGYM_IMAGES' = true ] || [ '$PREPULL_R2EGYM_IMAGES' = 1 ]; then
        PYTHONPATH='$REPO_ROOT' '$PYTHON_BIN' '$SCRIPT_DIR/prepull_images.py' \
          --train-data $train_data_escaped \
          --eval-data $eval_data_escaped \
          --mode pull \
          --retries '$PREPULL_R2EGYM_IMAGE_RETRIES' \
          --retry-sleep '$PREPULL_R2EGYM_IMAGE_RETRY_SLEEP'
      fi
      if command -v uv >/dev/null 2>&1; then
        export HOME='$MERGED_TMP_DIR/mini-swe-tool-cache-warmup-home'
        export UV_CACHE_DIR='$HARBOR_SHARED_UV_CACHE_HOST_DIR'
        mkdir -p \"\$HOME\"
        if [ '$HARBOR_MINI_SWE_AGENT_UV_OFFLINE' = 1 ]; then
          uv tool install --offline --cache-dir '$HARBOR_SHARED_UV_CACHE_HOST_DIR' '$HARBOR_MINI_SWE_AGENT_PACKAGE'
        else
          uv tool install --cache-dir '$HARBOR_SHARED_UV_CACHE_HOST_DIR' '$HARBOR_MINI_SWE_AGENT_PACKAGE'
        fi
        uv tool list
      else
        echo 'Missing required command: uv' >&2
        exit 1
      fi
      tail -f /dev/null"
  sleep 3
}

run_ray() {
  start_detached_client "ray_head" "$RAY_LOG" \
    srun --jobid "$MERGED_JOB_ID" --overlap --overcommit --immediate=10 -w "$MERGED_NODE" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
    bash -lc "set -euo pipefail
      mkdir -p '$HEAD_RAY_TMP_DIR'
      mkdir -p '$TRITON_CACHE_DIR'
      export TRITON_CACHE_DIR='$TRITON_CACHE_DIR'
      DEFAULT_SOCKET_IFNAME=\$(ip route get '$SOCKET_IFNAME_PEER_IP' 2>/dev/null | awk '{for (i=1; i<=NF; i++) if (\$i==\"dev\" && i<NF) {print \$(i+1); exit}}')
      if [ -z \"\$DEFAULT_SOCKET_IFNAME\" ]; then
        DEFAULT_SOCKET_IFNAME=\$(ip route show default 2>/dev/null | awk '{for (i=1; i<=NF; i++) if (\$i==\"dev\" && i<NF) {print \$(i+1); exit}}')
      fi
      if [ \"\${SKYRL_RESPECT_SOCKET_IFNAME:-0}\" != 1 ]; then
        export NCCL_SOCKET_IFNAME=\"\$DEFAULT_SOCKET_IFNAME\"
        export GLOO_SOCKET_IFNAME=\"\$DEFAULT_SOCKET_IFNAME\"
      else
        export NCCL_SOCKET_IFNAME=\"\${NCCL_SOCKET_IFNAME:-\$DEFAULT_SOCKET_IFNAME}\"
        export GLOO_SOCKET_IFNAME=\"\${GLOO_SOCKET_IFNAME:-\$DEFAULT_SOCKET_IFNAME}\"
      fi
      echo \"Using socket interfaces: NCCL_SOCKET_IFNAME=\$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=\$GLOO_SOCKET_IFNAME target=$SOCKET_IFNAME_PEER_IP\"
      '$RAY_BIN' stop -f >/dev/null 2>&1 || true
      '$RAY_BIN' start --head --disable-usage-stats --block --port '$RAY_PORT' --dashboard-host 0.0.0.0 --dashboard-port 8265 --node-ip-address '$MERGED_IP' --num-cpus '$HEAD_CPUS' --num-gpus '$HEAD_GPUS' --resources '{\"harbor_head\": 1}' --temp-dir '$HEAD_RAY_TMP_DIR'"

  for idx in "${!TRAINER_NODES[@]}"; do
    start_detached_client "ray_worker_$idx" "$RAY_LOG" \
      srun --jobid "${TRAINER_JOB_IDS[$idx]}" --overlap --overcommit --immediate=10 -w "${TRAINER_NODES[$idx]}" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
      bash -lc "set -euo pipefail
        mkdir -p '$TRAINER_RAY_TMP_DIR_ROOT/worker_$idx'
        mkdir -p '$TRITON_CACHE_DIR'
        export TRITON_CACHE_DIR='$TRITON_CACHE_DIR'
        '$RAY_BIN' stop -f >/dev/null 2>&1 || true
        NODE_IP=\$(hostname -I | tr ' ' '\n' | grep '^172\.27\.' | head -n1 || hostname -I | awk '{print \$1}')
        DEFAULT_SOCKET_IFNAME=\$(ip route get '$MERGED_IP' 2>/dev/null | awk '{for (i=1; i<=NF; i++) if (\$i==\"dev\" && i<NF) {print \$(i+1); exit}}')
        if [ -z \"\$DEFAULT_SOCKET_IFNAME\" ]; then
          DEFAULT_SOCKET_IFNAME=\$(ip route show default 2>/dev/null | awk '{for (i=1; i<=NF; i++) if (\$i==\"dev\" && i<NF) {print \$(i+1); exit}}')
        fi
        if [ \"\${SKYRL_RESPECT_SOCKET_IFNAME:-0}\" != 1 ]; then
          export NCCL_SOCKET_IFNAME=\"\$DEFAULT_SOCKET_IFNAME\"
          export GLOO_SOCKET_IFNAME=\"\$DEFAULT_SOCKET_IFNAME\"
        else
          export NCCL_SOCKET_IFNAME=\"\${NCCL_SOCKET_IFNAME:-\$DEFAULT_SOCKET_IFNAME}\"
          export GLOO_SOCKET_IFNAME=\"\${GLOO_SOCKET_IFNAME:-\$DEFAULT_SOCKET_IFNAME}\"
        fi
        echo \"Using socket interfaces: NCCL_SOCKET_IFNAME=\$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=\$GLOO_SOCKET_IFNAME target=$MERGED_IP\"
        '$RAY_BIN' start --disable-usage-stats --block --address '$MERGED_IP:$RAY_PORT' --node-ip-address \"\$NODE_IP\" --num-cpus '$TRAINER_CPUS' --num-gpus '$TRAINER_GPUS' --temp-dir '$TRAINER_RAY_TMP_DIR_ROOT/worker_$idx'"
  done
  sleep 10
}

run_rollout() {
  # Run on the rollout node. The helper only starts local vLLM server processes;
  # run_stages.sh owns node selection, ports, logging, and readiness checks.
  start_detached_client "rollout" "$ROLLOUT_LOG" \
    srun --jobid "$ROLLOUT_JOB_ID_EFFECTIVE" --overlap --overcommit --immediate=10 -w "$ROLLOUT_NODE_EFFECTIVE" --ntasks=1 --nodes=1 --cpus-per-task="$ROLLOUT_CPUS" --gres="gpu:$ROLLOUT_GPUS" \
    bash -lc "set -euo pipefail
      cd '$REPO_ROOT'
      export PATH='$PYTHON_BIN_DIR':\$HOME/.local/bin:\$PATH
      export PYTHON_BIN='$PYTHON_BIN'
      export RUN_NAME='$RUN_NAME'
      export LOG_DIR='$ROLLOUT_LOG_DIR'
      export SCRATCH_ROOT='$ROLLOUT_RUNTIME_SCRATCH_ROOT'
      export MODEL_PATH='$MODEL_PATH'
      export RAY_HEAD_IP='$MERGED_IP'
      export SOCKET_IFNAME_TARGET_IP='$SOCKET_IFNAME_PEER_IP'
      export ROLLOUT_SERVER_PORTS_CSV='$ROLLOUT_SERVER_PORTS_CSV'
      export ROLLOUT_GPU_GROUPS_SPEC='$ROLLOUT_GPU_GROUPS_SPEC'
      export ROLLOUT_TP_SIZE='$ROLLOUT_TP_SIZE'
      export ROLLOUT_ENFORCE_EAGER='$ROLLOUT_ENFORCE_EAGER'
      export VLLM_SERVER_MODULE='$VLLM_SERVER_MODULE'
      bash '$SCRIPT_DIR/start_rollout_servers.sh'"
  for port in "${ROLLOUT_PORTS[@]}"; do
    wait_for_http "http://$ROLLOUT_IP:$port/health" "rollout:$port" 900
  done
}

run_driver() {
  # Run on the head node after Ray and rollout servers are ready. The helper
  # only translates environment defaults into the SkyRL training command.
  local train_data_escaped=""
  local eval_data_escaped=""
  printf -v train_data_escaped '%q' "$TRAIN_DATA"
  printf -v eval_data_escaped '%q' "$EVAL_DATA"
  srun --jobid "$MERGED_JOB_ID" --overlap --overcommit --immediate=10 -w "$MERGED_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 --gres=gpu:0 \
    bash -lc "set -euo pipefail
      cd '$REPO_ROOT'
      export PATH='$PYTHON_BIN_DIR':\$HOME/.local/bin:\$PATH
      export PYTHON_BIN='$PYTHON_BIN'
      export RAY_ADDRESS='$MERGED_IP:$RAY_PORT'
      export RAY_HEAD_IP='$MERGED_IP'
      export ROLLOUT_HOST_IP='$ROLLOUT_IP'
      mkdir -p '$TRITON_CACHE_DIR'
      export TRITON_CACHE_DIR='$TRITON_CACHE_DIR'
      DEFAULT_SOCKET_IFNAME=\$(ip route get '$SOCKET_IFNAME_PEER_IP' 2>/dev/null | awk '{for (i=1; i<=NF; i++) if (\$i==\"dev\" && i<NF) {print \$(i+1); exit}}')
      if [ -z \"\$DEFAULT_SOCKET_IFNAME\" ]; then
        DEFAULT_SOCKET_IFNAME=\$(ip route show default 2>/dev/null | awk '{for (i=1; i<=NF; i++) if (\$i==\"dev\" && i<NF) {print \$(i+1); exit}}')
      fi
      if [ \"\${SKYRL_RESPECT_SOCKET_IFNAME:-0}\" != 1 ]; then
        export NCCL_SOCKET_IFNAME=\"\$DEFAULT_SOCKET_IFNAME\"
        export GLOO_SOCKET_IFNAME=\"\$DEFAULT_SOCKET_IFNAME\"
      else
        export NCCL_SOCKET_IFNAME=\"\${NCCL_SOCKET_IFNAME:-\$DEFAULT_SOCKET_IFNAME}\"
        export GLOO_SOCKET_IFNAME=\"\${GLOO_SOCKET_IFNAME:-\$DEFAULT_SOCKET_IFNAME}\"
      fi
      echo \"Using socket interfaces: NCCL_SOCKET_IFNAME=\$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=\$GLOO_SOCKET_IFNAME target=$SOCKET_IFNAME_PEER_IP\"
      export SKYRL_INFERENCE_ROUTER_PORT='$SKYRL_INFERENCE_ROUTER_PORT'
      export THUNDER_AGENT_ROUTER_PORT='$SKYRL_INFERENCE_ROUTER_PORT'
      export TRAIN_DATA=$train_data_escaped
      export EVAL_DATA=$eval_data_escaped
      export ROLLOUT_SERVER_PORTS_CSV='$ROLLOUT_SERVER_PORTS_CSV'
      export ROLLOUT_ENGINES='$ROLLOUT_ENGINES'
      export ROLLOUT_TP_SIZE='$ROLLOUT_TP_SIZE'
      export ROLLOUT_ENFORCE_EAGER='$ROLLOUT_ENFORCE_EAGER'
      export VLLM_SERVER_MODULE='$VLLM_SERVER_MODULE'
      export TRAIN_NUM_NODES=4
      export TRAIN_GPUS_PER_NODE=8
      export MODEL_PATH='$MODEL_PATH'
      export RUN_NAME_OVERRIDE='$RUN_NAME'
      export LOG_DIR_OVERRIDE='$LOG_DIR'
      export RUN_ARTIFACT_ROOT='$RUN_ARTIFACT_ROOT'
      export CKPT_ROOT_OVERRIDE='$CKPT_ROOT_OVERRIDE'
      export CKPT_INTERVAL='$CKPT_INTERVAL'
      export HF_SAVE_INTERVAL='$HF_SAVE_INTERVAL'
      export EXPORT_ROOT_OVERRIDE='$EXPORT_ROOT_OVERRIDE'
      export TRAINER_RESUME_MODE='$TRAINER_RESUME_MODE'
      export TRAINER_RESUME_PATH='$TRAINER_RESUME_PATH'
      export MAX_TRAIN_TASKS='$MAX_TRAIN_TASKS'
      export MAX_EVAL_TASKS='$MAX_EVAL_TASKS'
      export FULL_EPOCHS='$FULL_EPOCHS'
      export EVAL_INTERVAL_STEPS='$EVAL_INTERVAL_STEPS'
      export USE_KL_LOSS='$USE_KL_LOSS'
      export KL_LOSS_COEF='$KL_LOSS_COEF'
      export HARBOR_AGENT_MAX_TURNS='$HARBOR_AGENT_MAX_TURNS'
      export HARBOR_AGENT_TEMPERATURE='$HARBOR_AGENT_TEMPERATURE'
      export AGENT_TIMEOUT_SEC='$AGENT_TIMEOUT_SEC'
      export MINI_SWE_MODEL_TIMEOUT_SEC='$MINI_SWE_MODEL_TIMEOUT_SEC'
      export HARBOR_VERIFIER_TIMEOUT_MAX_ATTEMPTS='$HARBOR_VERIFIER_TIMEOUT_MAX_ATTEMPTS'
      export HARBOR_HARD_FAILURE_EXCEPTION_TYPES='$HARBOR_HARD_FAILURE_EXCEPTION_TYPES'
      export HARBOR_TASK_CIRCUIT_BREAKER_ENABLED='$HARBOR_TASK_CIRCUIT_BREAKER_ENABLED'
      export HARBOR_TASK_CIRCUIT_BREAKER_THRESHOLD='$HARBOR_TASK_CIRCUIT_BREAKER_THRESHOLD'
      export SKYRL_WORKER_NCCL_TIMEOUT_IN_S='$SKYRL_WORKER_NCCL_TIMEOUT_IN_S'
      export DOCKER_MODE='$DOCKER_MODE'
      export DOCKER_HOST='$DOCKER_HOST'
      export HARBOR_DOCKER_SHARED_NETWORK_NAME='$HARBOR_DOCKER_SHARED_NETWORK_NAME'
      export HARBOR_SHARED_UV_CACHE_HOST_DIR='$HARBOR_SHARED_UV_CACHE_HOST_DIR'
      export HARBOR_SHARED_UV_CACHE_ENV_DIR='$HARBOR_SHARED_UV_CACHE_ENV_DIR'
      export HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME='$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME'
      export HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME='$HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME'
      export HARBOR_SHARED_UV_PYTHON_HOST_DIR='$HARBOR_SHARED_UV_PYTHON_HOST_DIR'
      export HARBOR_SHARED_UV_PYTHON_ENV_DIR='$HARBOR_SHARED_UV_PYTHON_ENV_DIR'
      export HARBOR_MINI_SWE_AGENT_PACKAGE='$HARBOR_MINI_SWE_AGENT_PACKAGE'
      export HARBOR_MINI_SWE_AGENT_UV_OFFLINE='$HARBOR_MINI_SWE_AGENT_UV_OFFLINE'
      export THUNDERAGENT_WATCHDOG_ENABLED='$THUNDERAGENT_WATCHDOG_ENABLED'
      export RUN_PREFLIGHT_CHECKS='$RUN_PREFLIGHT_CHECKS'
      unset SKYRL_DISABLE_THUNDERAGENT
      stdbuf -oL -eL bash '$SCRIPT_DIR/run_trainer.sh' full" \
    >"$TRAIN_DRIVER_LOG" 2>&1
}

show_status() {
  echo "log_dir=$LOG_DIR"
  for stage in head ray_head ray_worker_0 ray_worker_1 ray_worker_2 ray_worker_3 rollout; do
    if stage_pid_is_running "$stage"; then
      echo "$stage RUNNING pid=$(cat "$(stage_pid_file "$stage")")"
    else
      echo "$stage STOPPED"
    fi
  done
  for port in "${ROLLOUT_PORTS[@]}"; do
    if curl -sf "http://$ROLLOUT_IP:$port/health" >/dev/null; then
      echo "rollout:$port HEALTHY"
    else
      echo "rollout:$port NOT_READY"
    fi
  done
}

cleanup_docker() {
  srun --jobid "$MERGED_JOB_ID" --overlap --overcommit --immediate=30 -w "$MERGED_NODE" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
    bash -lc "set -euo pipefail
      export PATH='$PYTHON_BIN_DIR':\$HOME/.local/bin:\$PATH
      export DOCKER_MODE='$DOCKER_MODE'
      export DOCKER_HOST='$DOCKER_HOST'
      export DATA_ROOT='$DATA_ROOT'
      if [ '$DOCKER_MODE' = rootful ] && ! docker info >/dev/null 2>&1; then
        if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1 && command -v setfacl >/dev/null 2>&1; then
          sudo -n setfacl -m u:'$USER':rw /var/run/docker.sock /run/docker.sock 2>/dev/null || true
        fi
      fi
      bash '$SCRIPT_DIR/cleanup_docker.sh'"
}

cleanup_stage() {
  local stage="$1"
  local pid_file=""
  local pid=""
  case "$stage" in
    rollout|head|ray_head|ray_worker_0|ray_worker_1|ray_worker_2|ray_worker_3)
      pid_file="$(stage_pid_file "$stage")"
      if [ -f "$pid_file" ]; then
        pid="$(cat "$pid_file" 2>/dev/null || true)"
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
      fi
      ;;
    ray)
      for s in ray_head ray_worker_0 ray_worker_1 ray_worker_2 ray_worker_3; do cleanup_stage "$s"; done
      ;;
    driver)
      pid_file="$(stage_pid_file driver)"
      if [ -f "$pid_file" ]; then
        pid="$(cat "$pid_file" 2>/dev/null || true)"
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
      fi
      ;;
    harbor_docker)
      cleanup_docker
      ;;
    all)
      cleanup_stage rollout
      cleanup_stage ray
      cleanup_stage head
      cleanup_stage driver
      cleanup_stage harbor_docker
      ;;
    *)
      echo "Unknown cleanup stage: $stage" >&2
      exit 1
      ;;
  esac
}

usage() {
  cat <<EOF
Usage:
  bash $0 cleanup-stage all
  bash $0 cleanup-stage harbor_docker
  bash $0 prepare
  bash $0 head
  bash $0 ray
  bash $0 rollout
  bash $0 status
  bash $0 driver
EOF
}

ensure_launch_requirements
print_banner

case "$ACTION" in
  prepare) run_prepare ;;
  head) run_head ;;
  ray) run_ray ;;
  rollout) run_rollout ;;
  status) show_status ;;
  driver) run_driver ;;
  cleanup-stage)
    [ -n "$ACTION_ARG" ] || {
      echo "cleanup-stage requires an argument" >&2
      exit 1
    }
    cleanup_stage "$ACTION_ARG"
    ;;
  *) usage >&2; exit 1 ;;
esac
