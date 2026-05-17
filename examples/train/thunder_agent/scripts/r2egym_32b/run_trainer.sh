#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# R2EGym 32B ThunderAgent Harbor training launcher.
#
# Benchmark variant: thunderagent_medium_hard_256_10epoch_no_preflight
# Spec summary:
#   - model: Qwen3-32B
#   - data: r2egym-train256-medium-hard-v1 / r2egym-eval64-medium-hard-v1
#   - trainer: 4 nodes x 8 GPUs = 32 GPUs, FSDP2
#   - rollout: 4 external vLLM servers, TP=2
#   - algorithm: GRPO with token-level off-policy correction
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
STAGE="${1:-full}"
if [ "$#" -gt 0 ]; then
  shift
fi

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

PYTHON_BIN="$(resolve_python_bin)"

# ---------------------------------------------------------------------------
# Data paths (adjust to your cluster)
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$HOME/data/harbor}"
TRAIN_DATA="${TRAIN_DATA:-['$DATA_ROOT/r2egym-train256-medium-hard-v1']}"
EVAL_DATA="${EVAL_DATA:-['$DATA_ROOT/r2egym-eval64-medium-hard-v1']}"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-32B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-32B}"
MODEL_REPO_ID="${MODEL_REPO_ID:-Qwen/Qwen3-32B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-$REPO_ROOT/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2}"

# ---------------------------------------------------------------------------
# Training topology
# ---------------------------------------------------------------------------
TRAIN_NUM_NODES="${TRAIN_NUM_NODES:-4}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-8}"
TRAINING_WORLD_SIZE=$((TRAIN_NUM_NODES * TRAIN_GPUS_PER_NODE))

# ---------------------------------------------------------------------------
# Rollout topology
# ---------------------------------------------------------------------------
ROLLOUT_ENGINES="${ROLLOUT_ENGINES:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
# ROLLOUT_SERVER_URLS can be set directly as a Python list literal, e.g.
#   '["http://1.2.3.4:18000","http://1.2.3.4:18001"]'
# OR constructed automatically from ROLLOUT_HOST_IP + ROLLOUT_SERVER_PORTS_CSV
# (same convention as run_harbor_benchmark.sh / run_harbor_fully_async.sh).
ROLLOUT_SERVER_URLS="${ROLLOUT_SERVER_URLS:-}"
ROLLOUT_HOST_IP="${ROLLOUT_HOST_IP:-}"
ROLLOUT_SERVER_PORTS_CSV="${ROLLOUT_SERVER_PORTS_CSV:-18000,18001,18002,18003}"

# Auto-build ROLLOUT_SERVER_URLS from host+ports if not already set.
if [ -z "$ROLLOUT_SERVER_URLS" ] && [ -n "$ROLLOUT_HOST_IP" ]; then
  _urls=()
  IFS=',' read -r -a _ports <<<"$ROLLOUT_SERVER_PORTS_CSV"
  for _p in "${_ports[@]}"; do
    _urls+=('"'"http://${ROLLOUT_HOST_IP}:${_p}"'"')
  done
  ROLLOUT_SERVER_URLS="[$(IFS=,; echo "${_urls[*]}")]"
  unset _urls _ports _p
fi

# EXTERNAL_PROXY_URL: when run_harbor_benchmark.sh already started a TA proxy
# on the merged/head node, pass its URL so SkyRL reuses it instead of starting
# an embedded ThunderAgentRouter. Also accepts THUNDERAGENT_URL (same variable
# name used by run_harbor_fully_async.sh).
EXTERNAL_PROXY_URL="${EXTERNAL_PROXY_URL:-${THUNDERAGENT_URL:-}}"
# Only derive a proxy URL when this recipe is explicitly told that an external
# ThunderAgent router was started. Otherwise the trainer creates its embedded
# router over the external rollout servers.
if [ "${USE_EXTERNAL_THUNDERAGENT_PROXY:-0}" = "1" ] && [ -z "$EXTERNAL_PROXY_URL" ] && [ -n "${RAY_HEAD_IP:-}" ]; then
  EXTERNAL_PROXY_URL="http://${RAY_HEAD_IP}:${SKYRL_INFERENCE_ROUTER_PORT:-${THUNDER_AGENT_ROUTER_PORT:-8080}}"
fi

if [ "$STAGE" = "url_test" ]; then
  echo "ROLLOUT_SERVER_URLS=$ROLLOUT_SERVER_URLS"
  echo "EXTERNAL_PROXY_URL=$EXTERNAL_PROXY_URL"
  echo "THUNDERAGENT_URL=${THUNDERAGENT_URL:-}"
  exit 0
fi

# ---------------------------------------------------------------------------
# ThunderAgent
# ---------------------------------------------------------------------------
THUNDER_AGENT_MODE="${THUNDER_AGENT_MODE:-tr}"
THUNDER_AGENT_PROFILE_ENABLED="${THUNDER_AGENT_PROFILE_ENABLED:-true}"
THUNDER_AGENT_METRICS_ENABLED="${THUNDER_AGENT_METRICS_ENABLED:-true}"

# ---------------------------------------------------------------------------
# Harbor runtime
# ---------------------------------------------------------------------------
HARBOR_AGENT_MAX_TURNS="${HARBOR_AGENT_MAX_TURNS:-25}"
HARBOR_AGENT_TEMPERATURE="${HARBOR_AGENT_TEMPERATURE:-0.3}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-9000}"
MINI_SWE_MODEL_TIMEOUT_SEC="${MINI_SWE_MODEL_TIMEOUT_SEC:-1200}"
DOCKER_MODE="${DOCKER_MODE:-rootful}"

# ---------------------------------------------------------------------------
# Logging / artifacts
# ---------------------------------------------------------------------------
RUN_NAME="${RUN_NAME_OVERRIDE:-r2egym-ta-mediumhard256-10epoch-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/tmp_logs}"
LOG_DIR="${LOG_DIR_OVERRIDE:-$LOG_ROOT/$RUN_NAME}"
TENSORBOARD_DIR="$LOG_DIR/tensorboard"
RUN_ARTIFACT_ROOT="${RUN_ARTIFACT_ROOT:-/scratch/$USER/harbor_run_artifacts}"
RUN_ARTIFACT_DIR="$RUN_ARTIFACT_ROOT/$RUN_NAME"
TRIALS_DIR="$RUN_ARTIFACT_DIR/trials_run"
CKPT_ROOT_OVERRIDE="${CKPT_ROOT_OVERRIDE:-}"
EXPORT_ROOT_OVERRIDE="${EXPORT_ROOT_OVERRIDE:-}"

if [ -n "$CKPT_ROOT_OVERRIDE" ]; then
  CKPTS_DIR="$CKPT_ROOT_OVERRIDE/$RUN_NAME/ckpts"
else
  CKPTS_DIR="$RUN_ARTIFACT_DIR/ckpts"
fi
if [ -n "$EXPORT_ROOT_OVERRIDE" ]; then
  EXPORTS_DIR="$EXPORT_ROOT_OVERRIDE/$RUN_NAME/exports"
else
  EXPORTS_DIR="$RUN_ARTIFACT_DIR/exports"
fi

mkdir -p "$TRIALS_DIR" "$CKPTS_DIR" "$EXPORTS_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Algorithm hyperparameters
# ---------------------------------------------------------------------------
FULL_TRAIN_BATCH_SIZE="${FULL_TRAIN_BATCH_SIZE:-64}"
FULL_POLICY_MINI_BATCH_SIZE="${FULL_POLICY_MINI_BATCH_SIZE:-64}"
FULL_MICRO_FORWARD_BATCH_SIZE_PER_GPU="${FULL_MICRO_FORWARD_BATCH_SIZE_PER_GPU:-4}"
FULL_MICRO_TRAIN_BATCH_SIZE_PER_GPU="${FULL_MICRO_TRAIN_BATCH_SIZE_PER_GPU:-4}"
FULL_EPOCHS="${FULL_EPOCHS:-10}"
FULL_N_SAMPLES="${FULL_N_SAMPLES:-4}"
FULL_NUM_PARALLEL_GENERATION_WORKERS="${FULL_NUM_PARALLEL_GENERATION_WORKERS:-64}"
FULL_MAX_CONCURRENCY="${FULL_MAX_CONCURRENCY:-256}"
FULL_TRAJ_PER_SEC="${FULL_TRAJ_PER_SEC:-2}"
FULL_MAX_STALENESS_STEPS="${FULL_MAX_STALENESS_STEPS:-2}"

USE_KL_LOSS="${USE_KL_LOSS:-false}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
TIS_TYPE="${TIS_TYPE:-token}"
TIS_IMP_RATIO_CAP="${TIS_IMP_RATIO_CAP:-2.0}"
LOSS_REDUCTION="${LOSS_REDUCTION:-seq_mean_token_sum_norm}"
GRPO_NORM_BY_STD="${GRPO_NORM_BY_STD:-false}"
APPLY_OVERLONG_FILTERING="${APPLY_OVERLONG_FILTERING:-true}"
TRAIN_MAX_SEQ_LEN="${TRAIN_MAX_SEQ_LEN:-6144}"

EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-4}"
CKPT_INTERVAL="${CKPT_INTERVAL:-4}"
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:--1}"

# ---------------------------------------------------------------------------
# Stage-specific overrides
# ---------------------------------------------------------------------------
case "$STAGE" in
  smoke)
    MAX_TRAIN_TASKS=8
    MAX_EVAL_TASKS=4
    N_SAMPLES=2
    EVAL_N_SAMPLES=1
    TRAIN_BATCH_SIZE=8
    MINI_BATCH_SIZE=8
    MICRO_FORWARD_BATCH_SIZE_PER_GPU=1
    MICRO_TRAIN_BATCH_SIZE_PER_GPU=1
    EPOCHS=1
    EVAL_INTERVAL=0
    TRAJ_PER_SEC=1
    MAX_CONCURRENCY=2
    NUM_PARALLEL_GENERATION_WORKERS=8
    ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    ;;
  pilot)
    MAX_TRAIN_TASKS=64
    MAX_EVAL_TASKS=10
    N_SAMPLES=2
    EVAL_N_SAMPLES=1
    TRAIN_BATCH_SIZE="$TRAINING_WORLD_SIZE"
    MINI_BATCH_SIZE="$TRAINING_WORLD_SIZE"
    MICRO_FORWARD_BATCH_SIZE_PER_GPU=1
    MICRO_TRAIN_BATCH_SIZE_PER_GPU=1
    EPOCHS=1
    EVAL_INTERVAL=20
    TRAJ_PER_SEC=1
    MAX_CONCURRENCY=4
    NUM_PARALLEL_GENERATION_WORKERS="$TRAINING_WORLD_SIZE"
    ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    ;;
  full)
    MAX_TRAIN_TASKS=256
    MAX_EVAL_TASKS=64
    N_SAMPLES="$FULL_N_SAMPLES"
    EVAL_N_SAMPLES=1
    TRAIN_BATCH_SIZE="$FULL_TRAIN_BATCH_SIZE"
    MINI_BATCH_SIZE="$FULL_POLICY_MINI_BATCH_SIZE"
    MICRO_FORWARD_BATCH_SIZE_PER_GPU="$FULL_MICRO_FORWARD_BATCH_SIZE_PER_GPU"
    MICRO_TRAIN_BATCH_SIZE_PER_GPU="$FULL_MICRO_TRAIN_BATCH_SIZE_PER_GPU"
    EPOCHS="$FULL_EPOCHS"
    EVAL_INTERVAL="$EVAL_INTERVAL_STEPS"
    TRAJ_PER_SEC="$FULL_TRAJ_PER_SEC"
    MAX_CONCURRENCY="$FULL_MAX_CONCURRENCY"
    NUM_PARALLEL_GENERATION_WORKERS="$FULL_NUM_PARALLEL_GENERATION_WORKERS"
    ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    ;;
  *)
    echo "Usage: $0 {smoke|pilot|full}"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export _SKYRL_USE_NEW_INFERENCE=1
export SKYRL_INFERENCE_ROUTER_PORT="${SKYRL_INFERENCE_ROUTER_PORT:-8080}"
# main_thunder_agent.py reads THUNDER_AGENT_ROUTER_PORT for the embedded router.
export THUNDER_AGENT_ROUTER_PORT="${THUNDER_AGENT_ROUTER_PORT:-$SKYRL_INFERENCE_ROUTER_PORT}"
export RAY_ADDRESS="${RAY_ADDRESS:-}"
export TENSORBOARD_DIR
export SKYRL_TRIALS_ROOT="$TRIALS_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ -z "$ROLLOUT_SERVER_URLS" ]; then
  echo "ROLLOUT_SERVER_URLS is required. Set it to the list of external vLLM server URLs." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
ENTRYPOINT="examples.train.thunder_agent.main_harbor_thunder_agent"

EXTRA_ARGS=()
if [ -n "$EXTERNAL_PROXY_URL" ]; then
  EXTRA_ARGS+=("generator.inference_engine.external_proxy_url=$EXTERNAL_PROXY_URL")
fi
if [ -n "${TRAINER_RESUME_PATH:-}" ]; then
  EXTRA_ARGS+=("trainer.resume_path=$TRAINER_RESUME_PATH")
fi

"$PYTHON_BIN" -m "$ENTRYPOINT" \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$EVAL_DATA" \
  max_train_tasks="$MAX_TRAIN_TASKS" \
  max_eval_tasks="$MAX_EVAL_TASKS" \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.inference_engine.served_model_name="$MODEL_NAME" \
  harbor_trial_config.trials_dir="$TRIALS_DIR" \
  trainer.export_path="$EXPORTS_DIR" \
  trainer.ckpt_path="$CKPTS_DIR" \
  trainer.log_path="$LOG_DIR" \
  trainer.strategy=fsdp2 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.off_policy_correction.tis_ratio_type="$TIS_TYPE" \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high="$TIS_IMP_RATIO_CAP" \
  trainer.algorithm.loss_reduction="$LOSS_REDUCTION" \
  trainer.algorithm.grpo_norm_by_std="$GRPO_NORM_BY_STD" \
  trainer.algorithm.use_kl_loss="$USE_KL_LOSS" \
  trainer.algorithm.kl_loss_coef="$KL_LOSS_COEF" \
  trainer.algorithm.max_seq_len="$TRAIN_MAX_SEQ_LEN" \
  trainer.fully_async.max_staleness_steps="$FULL_MAX_STALENESS_STEPS" \
  trainer.fully_async.num_parallel_generation_workers="$NUM_PARALLEL_GENERATION_WORKERS" \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.placement.policy_num_nodes="$TRAIN_NUM_NODES" \
  trainer.placement.policy_num_gpus_per_node="$TRAIN_GPUS_PER_NODE" \
  trainer.placement.ref_num_nodes="$TRAIN_NUM_NODES" \
  trainer.placement.ref_num_gpus_per_node="$TRAIN_GPUS_PER_NODE" \
  trainer.critic.model.path=null \
  trainer.epochs="$EPOCHS" \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval="$EVAL_INTERVAL" \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$MINI_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_FORWARD_BATCH_SIZE_PER_GPU" \
  trainer.micro_train_batch_size_per_gpu="$MICRO_TRAIN_BATCH_SIZE_PER_GPU" \
  trainer.flash_attn=true \
  trainer.policy.record_memory=true \
  trainer.use_sample_packing=false \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.hf_save_interval="$HF_SAVE_INTERVAL" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.n_samples_per_prompt="$N_SAMPLES" \
  generator.eval_n_samples_per_prompt="$EVAL_N_SAMPLES" \
  generator.apply_overlong_filtering="$APPLY_OVERLONG_FILTERING" \
  generator.inference_engine.num_engines="$ROLLOUT_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$ROLLOUT_TP_SIZE" \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.remote_urls="$ROLLOUT_SERVER_URLS" \
  generator.inference_engine.external_server_urls="$ROLLOUT_SERVER_URLS" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  generator.inference_engine.engine_init_kwargs.chat_template="$CHAT_TEMPLATE_PATH" \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second="$TRAJ_PER_SEC" \
  generator.rate_limit.max_concurrency="$MAX_CONCURRENCY" \
  generator.inference_engine.thunder_agent_mode="$THUNDER_AGENT_MODE" \
  generator.inference_engine.thunder_agent_profile_enabled="$THUNDER_AGENT_PROFILE_ENABLED" \
  generator.inference_engine.thunder_agent_metrics_enabled="$THUNDER_AGENT_METRICS_ENABLED" \
  harbor_trial_config.environment.type=docker \
  harbor_trial_config.environment.override_cpus=2 \
  harbor_trial_config.environment.override_memory_mb=4096 \
  harbor_trial_config.environment.override_storage_mb=4096 \
  harbor_trial_config.environment.kwargs.auto_stop_interval_mins=null \
  harbor_trial_config.agent.override_timeout_sec="$AGENT_TIMEOUT_SEC" \
  harbor_trial_config.agent.kwargs.max_turns="$HARBOR_AGENT_MAX_TURNS" \
  harbor_trial_config.agent.kwargs.temperature="$HARBOR_AGENT_TEMPERATURE" \
  harbor_trial_config.agent.kwargs.enable_summarize=false \
  harbor_trial_config.agent.kwargs.record_terminal_session=false \
  harbor_trial_config.agent.kwargs.store_all_messages=true \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.timeout="$MINI_SWE_MODEL_TIMEOUT_SEC" \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking=false \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.include_reasoning=false \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_MODEL_LEN" \
  trainer.logger="['tensorboard','console']" \
  trainer.project_name=harbor \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode="${TRAINER_RESUME_MODE:-latest}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
