#!/usr/bin/env bash
# mini-swe-agent on CodeContests through skycap: Harbor runs the agent inside its Daytona sandbox,
# and the agent calls the model from there.
#
# So each skycap server's harness routes are exposed to the sandboxes: at SKYCAP_EXTERNAL_HOST
# when they can route to this node, otherwise through a Cloudflare quick tunnel (see the README).
#
# Defaults are a small synchronous run on one GPU that overfits NUM_PROMPTS tasks (pick them with
# TASKS=a,b,...), one step per epoch (train_batch_size = NUM_PROMPTS):
#   TRAIN_PATHS=all|final   which captured paths train (skycap.train_paths; default all)
#   AGENT=<harbor agent>    another installed agent instead of mini-swe-agent
#
#   export DAYTONA_API_KEY=... WANDB_API_KEY=...   # WANDB optional: console logging without it
#   bash examples/train_integrations/harbor_skycap/run_codecontests_mini_swe_agent.sh [overrides...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$REPO/examples/train_integrations/harbor_skycap"
cd "$REPO"

: "${DAYTONA_API_KEY:?export DAYTONA_API_KEY}"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
AGENT="${AGENT:-mini-swe-agent}"
MINI_SWE_AGENT_VERSION="${MINI_SWE_AGENT_VERSION:-2.4.6}"
MINI_SWE_AGENT_CONFIG="${MINI_SWE_AGENT_CONFIG:-$HERE/mini_swe_agent.yaml}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-3.0e-6}"
# LR warms up from 0: a full-size first step at a higher LR broke the tool-call format.
WARMUP_STEPS="${WARMUP_STEPS:-5}"
TRAIN_PATHS="${TRAIN_PATHS:-all}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
NUM_GPUS="${NUM_GPUS:-1}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-1200}"
SANDBOX_MEMORY_MB="${SANDBOX_MEMORY_MB:-4096}"
# Checkpoints: every CKPT_INTERVAL steps (-1: none), keeping the newest MAX_CKPTS_TO_KEEP (-1: all);
# RESUME_MODE=latest continues from the newest checkpoint in the run's ckpts/ (a fresh run starts from scratch).
CKPT_INTERVAL="${CKPT_INTERVAL:--1}"
MAX_CKPTS_TO_KEEP="${MAX_CKPTS_TO_KEEP:--1}"
RESUME_MODE="${RESUME_MODE:-none}"
SKYCAP_EXTERNAL_HOST="${SKYCAP_EXTERNAL_HOST:-}"
DATASET="${DATASET:-open-thoughts/CodeContests}"
DATA_ROOT="${DATA_ROOT:-/tmp/harbor/data}"
TASKS_DIR="$DATA_ROOT/$(basename "$DATASET")"

EXPERIMENT="${EXPERIMENT:-harbor-skycap-$AGENT-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')}"
RUNS_ROOT="${RUNS_ROOT:-/tmp/harbor/runs}"
RUN_DIR="$RUNS_ROOT/$EXPERIMENT"
SUBSET_DIR="$RUN_DIR/tasks"

if [[ -n "${WANDB_API_KEY:-}" ]]; then LOGGER="${LOGGER:-wandb}"; else LOGGER="${LOGGER:-console}"; fi

if [[ ! -d "$TASKS_DIR" ]] || [[ -z "$(ls -A "$TASKS_DIR" 2>/dev/null)" ]]; then
  echo "==> preparing $DATASET into $TASKS_DIR"
  mkdir -p "$DATA_ROOT"
  uv run --isolated --extra harbor python examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset "$DATASET" --output_dir "$TASKS_DIR"
fi

# The first NUM_PROMPTS tasks, or the comma-separated TASKS.
mkdir -p "$SUBSET_DIR"
if [[ -n "${TASKS:-}" ]]; then
  IFS=, read -r -a tasks <<< "$TASKS"
  tasks=("${tasks[@]/#/$TASKS_DIR/}")
else
  mapfile -t tasks < <(find "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | sort | head -n "$NUM_PROMPTS")
fi
for task in "${tasks[@]}"; do
  [[ -d "$task" ]] || { echo "no task at $task" >&2; exit 1; }
  ln -sfn "$task" "$SUBSET_DIR/$(basename "$task")"
done
NUM_PROMPTS="${#tasks[@]}"
echo "==> $NUM_PROMPTS tasks x $GROUP_SIZE samples, $EPOCHS epochs, agent $AGENT, model $MODEL"

AGENT_ARGS=(harbor_trial_config.agent.name="$AGENT")
if [[ "$AGENT" == mini-swe-agent ]]; then
  AGENT_ARGS+=(
    harbor_trial_config.agent.kwargs.version="$MINI_SWE_AGENT_VERSION"
    harbor_trial_config.agent.kwargs.config_file="$MINI_SWE_AGENT_CONFIG"
  )
fi
EXPOSURE_ARGS=()
if [[ -n "$SKYCAP_EXTERNAL_HOST" ]]; then
  EXPOSURE_ARGS+=(skycap.exposure.type=external_host skycap.exposure.host="$SKYCAP_EXTERNAL_HOST")
else
  EXPOSURE_ARGS+=(skycap.exposure.type=cloudflare)
fi

# Qwen3.5's Gated DeltaNet backward goes through FLA's kernel-backend dispatch, which picks TileLang;
# TileLang JIT-compiles with the pip nvcc (13.4), whose CUDA headers (nvidia-cuda-cccl 13.3) it rejects
# as incompatible. Without the dispatch, FLA runs its own Triton kernels.
export FLA_DISABLE_BACKEND_DISPATCH="${FLA_DISABLE_BACKEND_DISPATCH:-1}"

# A private Ray cluster for the run, even on a node that already has one. Its task-event
# export to a dashboard aggregator is off: this cluster runs without a dashboard.
export RAY_ADDRESS=local
export RAY_enable_ray_event=0
export RAY_task_events_report_interval_ms=0
unset RAY_enable_core_worker_ray_event_to_aggregator RAY_enable_export_api_write_config
unset RAY_DASHBOARD_AGGREGATOR_AGENT_EVENTS_EXPORT_ADDR RAY_DASHBOARD_AGGREGATOR_AGENT_PUBLISHER_MAX_RETRIES

echo "==> experiment $EXPERIMENT in $RUN_DIR"
uv run --isolated --extra fsdp --extra harbor --extra skycap \
  -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap \
  data.train_data="['$SUBSET_DIR']" \
  trainer.policy.model.path="$MODEL" \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  trainer.project_name=harbor-skycap \
  trainer.run_name="$EXPERIMENT" \
  trainer.logger="$LOGGER" \
  trainer.export_path="$RUN_DIR/exports" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.max_ckpts_to_keep="$MAX_CKPTS_TO_KEEP" \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode="$RESUME_MODE" \
  skycap.record_dir="$RUN_DIR/skycap" \
  skycap.num_servers=1 \
  skycap.train_paths="$TRAIN_PATHS" \
  "${EXPOSURE_ARGS[@]}" \
  "${AGENT_ARGS[@]}" \
  harbor_trial_config.trials_dir="$RUN_DIR/trials" \
  harbor_trial_config.environment.type=daytona \
  harbor_trial_config.environment.override_cpus=1 \
  harbor_trial_config.environment.override_memory_mb="$SANDBOX_MEMORY_MB" \
  harbor_trial_config.environment.override_storage_mb=5120 \
  harbor_trial_config.agent.override_timeout_sec="$AGENT_TIMEOUT" \
  trainer.epochs="$EPOCHS" \
  trainer.train_batch_size="$NUM_PROMPTS" \
  trainer.policy_mini_batch_size="$NUM_PROMPTS" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  generator.n_samples_per_prompt="$GROUP_SIZE" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.off_policy_correction.tis_ratio_type=token \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.algorithm.temperature="$TEMPERATURE" \
  trainer.policy.optimizer_config.lr="$LR" \
  trainer.policy.optimizer_config.num_warmup_steps="$WARMUP_STEPS" \
  trainer.strategy=fsdp \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  trainer.remove_microbatch_padding=false \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.sampling_params.temperature="$TEMPERATURE" \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=false \
  generator.apply_overlong_filtering=true \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=2 \
  generator.rate_limit.max_concurrency=$((NUM_PROMPTS * GROUP_SIZE)) \
  "$@" 2>&1 | tee -a "$RUN_DIR/run.log"

echo
echo "==> done: $EXPERIMENT"
echo "    skycap records: $RUN_DIR/skycap"
echo "    Harbor trials:  $RUN_DIR/trials"
echo "    log:            $RUN_DIR/run.log"
