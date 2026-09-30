#!/usr/bin/env bash
# Harbor training on CodeContests, sandboxed on Daytona or Modal, through skycap or through
# the sibling `harbor` integration (the baseline), to compare the two.
#
#   bash examples/train_integrations/harbor_skycap/run_codecontests.sh [extra overrides...]
#
# The defaults are the sibling recipe, examples/train_integrations/harbor/run_codecontest.sh:
# Qwen3-8B, 32k context, 32 prompts x 8 samples per step, GRPO with token-mean loss and TIS,
# overlong filtering, 8 GPUs colocated with 4 vLLM engines at TP 2. The knobs below scale it down.
#
# Does everything: reads the sandbox credentials, prepares the tasks under /tmp/harbor/data
# (skipped when they are already there), and trains EPOCHS passes over the tasks. Each run
# gets a fresh uuid name, so its trials, skycap records, checkpoints and logs all land in
# their own /tmp/harbor/runs/<name>.
#
#   GENERATOR=skycap|baseline  which integration generates the rollouts (default skycap)
#   NUM_PROMPTS, GROUP_SIZE    prompts per step and samples per prompt (default 32 x 8)
#   TASKS=a,b                  task names to run instead of the first NUM_PROMPTS
#   EPOCHS, LR                 passes over the tasks, learning rate (default 1, 1e-6)
#   TEMPERATURE=0              greedy, so a baseline run and a skycap run can be diffed
#                              token for token with compare_runs.py
#   DETERMINISTIC=1            one trial at a time and no prefix caching, so every request runs
#                              alone and greedy decoding can't flip on batch-dependent numerics
#   LOGGER=console|wandb       W&B (project harbor-compare, run named after the experiment) by
#                              default when a key is found, else the console
#   SANDBOX=daytona|modal      where Harbor runs each trial's sandbox (default daytona)
#   STRATEGY=fsdp|megatron     the training backend (default fsdp); megatron takes MEGATRON_TP,
#                              MEGATRON_PP, MEGATRON_EP and MEGATRON_ETP, and OPTIMIZER_OFFLOAD=1
#                              (the default) keeps the optimizer state on the CPU, which an MoE on
#                              one node needs: expert parameters aren't sharded across data parallel
#   R3=1                       rollout routing replay for an MoE model: vLLM returns each token's
#                              routed experts and Megatron replays them (needs STRATEGY=megatron)
#
# Needs: the GPUs, and the sandbox's credentials in a file: `DAYTONA_API_KEY=...`
# (DAYTONA_KEY_FILE) or `MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...` (MODAL_KEY_FILE). For W&B,
# WANDB_API_KEY, or a file holding `WANDB_API_KEY=...` (WANDB_KEY_FILE).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

#-----------------------
# Knobs
#-----------------------
SANDBOX="${SANDBOX:-daytona}"
DAYTONA_KEY_FILE="${DAYTONA_KEY_FILE:-$HOME/default/daytona.key}"
MODAL_KEY_FILE="${MODAL_KEY_FILE:-$HOME/default/model_key.key}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/default/wandb_key.key}"
GENERATOR="${GENERATOR:-skycap}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
# Keeps thinking in history, so a multi-turn rollout stays append-only (the sibling recipe's choice).
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$REPO/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TASKS="${TASKS:-}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1.0e-6}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DETERMINISTIC="${DETERMINISTIC:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_ENGINES="${NUM_ENGINES:-4}"
TP_SIZE="${TP_SIZE:-2}"
SKYCAP_SERVERS="${SKYCAP_SERVERS:-2}"
# vLLM's share of each GPU. It wakes up next to the trainer's weights after each step, so a
# large model colocated with Megatron needs less than the recipe's 0.8 (0.6 for Qwen3-30B-A3B).
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
DATASET="${DATASET:-open-thoughts/CodeContests}"
STRATEGY="${STRATEGY:-fsdp}"
MEGATRON_TP="${MEGATRON_TP:-1}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_EP="${MEGATRON_EP:-8}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"
R3="${R3:-0}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-1}"

DATA_ROOT="/tmp/harbor/data"
TASKS_DIR="$DATA_ROOT/$(basename "$DATASET")"

EXPERIMENT="harbor-$GENERATOR-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
RUN_DIR="/tmp/harbor/runs/$EXPERIMENT"
SUBSET_DIR="$RUN_DIR/tasks"

case "$GENERATOR" in
  skycap)
    ENTRYPOINT=examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap
    EXTRAS=(--extra harbor --extra skycap)
    # Each row is already a complete path; merging could fuse two paths. Thinking stays in
    # history the way it does for the baseline because skycap answers with raw content
    # (skycap.use_raw_content, on by default): see SkycapConfig.
    ARM_ARGS=(
      skycap.record_dir="$RUN_DIR/skycap"
      skycap.num_servers="$SKYCAP_SERVERS"
      generator.merge_stepwise_output=false
    )
    ;;
  baseline)
    ENTRYPOINT=examples.train_integrations.harbor.entrypoints.main_harbor
    EXTRAS=(--extra harbor)
    # Its recommended setting: per-turn rows merged back into whole sequences where they align.
    ARM_ARGS=(generator.merge_stepwise_output=true)
    ;;
  *)
    echo "GENERATOR must be skycap or baseline, got $GENERATOR" >&2
    exit 1
    ;;
esac
case "$STRATEGY" in
  fsdp) STRATEGY_ARGS=() ;;
  megatron)
    STRATEGY_ARGS=(
      trainer.policy.megatron_config.tensor_model_parallel_size="$MEGATRON_TP"
      trainer.policy.megatron_config.pipeline_model_parallel_size="$MEGATRON_PP"
      trainer.policy.megatron_config.expert_model_parallel_size="$MEGATRON_EP"
      trainer.policy.megatron_config.expert_tensor_parallel_size="$MEGATRON_ETP"
      trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full
      trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform
      trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1
    )
    if [[ "$OPTIMIZER_OFFLOAD" == 1 ]]; then
      STRATEGY_ARGS+=(
        trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true
        trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0
        trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true
        trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=true
      )
    fi
    ;;
  *)
    echo "STRATEGY must be fsdp or megatron, got $STRATEGY" >&2
    exit 1
    ;;
esac
if [[ "$R3" == 1 ]]; then
  [[ "$STRATEGY" == megatron ]] || { echo "R3=1 needs STRATEGY=megatron" >&2; exit 1; }
  # vLLM's mp backend, as SkyRL requires for R3 (the ray backend can hang).
  STRATEGY_ARGS+=(
    trainer.policy.megatron_config.moe_enable_routing_replay=true
    generator.inference_engine.enable_return_routed_experts=true
    generator.inference_engine.distributed_executor_backend=mp
  )
fi
if [[ "$DETERMINISTIC" == 1 ]]; then
  ARM_ARGS+=(
    generator.rate_limit.max_concurrency=1
    generator.inference_engine.engine_init_kwargs.enable_prefix_caching=false
  )
fi

#-----------------------
# Credentials
#-----------------------
case "$SANDBOX" in
  daytona) KEY_FILE="$DAYTONA_KEY_FILE" ;;
  modal) KEY_FILE="$MODAL_KEY_FILE" ;;
  *)
    echo "SANDBOX must be daytona or modal, got $SANDBOX" >&2
    exit 1
    ;;
esac
if [[ ! -f "$KEY_FILE" ]]; then
  echo "no $SANDBOX credentials at $KEY_FILE" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$KEY_FILE"
set +a
if [[ "$SANDBOX" == daytona ]]; then
  : "${DAYTONA_API_KEY:?$KEY_FILE must set DAYTONA_API_KEY}"
else
  : "${MODAL_TOKEN_ID:?$KEY_FILE must set MODAL_TOKEN_ID}"
  : "${MODAL_TOKEN_SECRET:?$KEY_FILE must set MODAL_TOKEN_SECRET}"
fi

# SkyRL reads the W&B key from the environment only.
if [[ -z "${WANDB_API_KEY:-}" && -f "$WANDB_KEY_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$WANDB_KEY_FILE"
  set +a
fi
LOGGER="${LOGGER:-$([[ -n "${WANDB_API_KEY:-}" ]] && echo wandb || echo console)}"
if [[ "$LOGGER" == wandb ]]; then
  : "${WANDB_API_KEY:?LOGGER=wandb needs WANDB_API_KEY or WANDB_KEY_FILE}"
  export WANDB_API_KEY
fi

#-----------------------
# Data: download and extract once, then this run's tasks (TASKS, or the first NUM_PROMPTS)
#-----------------------
if [[ -d "$TASKS_DIR" ]] && [[ -n "$(ls -A "$TASKS_DIR" 2>/dev/null)" ]]; then
  echo "==> tasks already at $TASKS_DIR"
else
  echo "==> preparing $DATASET into $TASKS_DIR"
  mkdir -p "$DATA_ROOT"
  uv run --isolated --extra harbor python examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset "$DATASET" --output_dir "$TASKS_DIR"
fi
mkdir -p "$SUBSET_DIR"
if [[ -n "$TASKS" ]]; then
  IFS=, read -r -a tasks <<< "$TASKS"
  tasks=("${tasks[@]/#/$TASKS_DIR/}")
else
  mapfile -t tasks < <(find "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
  tasks=("${tasks[@]:0:$NUM_PROMPTS}")
fi
for task in "${tasks[@]}"; do
  [[ -d "$task" ]] || { echo "no task at $task" >&2; exit 1; }
  ln -s "$task" "$SUBSET_DIR/$(basename "$task")"
done
NUM_PROMPTS="${#tasks[@]}"
echo "==> $NUM_PROMPTS tasks x $GROUP_SIZE samples in $SUBSET_DIR"

#-----------------------
# Ray: a fresh local cluster for this run
#-----------------------
# A workspace may already run a Ray cluster of another version, and export settings
# (resource overrides, event export) that a local one can't use. Start clean.
export RAY_ADDRESS=local
if [[ -n "${RAY_OVERRIDE_RESOURCES:-}" ]]; then
  RAY_OVERRIDE_RESOURCES="$(python3 -c '
import json, os
resources = json.loads(os.environ["RAY_OVERRIDE_RESOURCES"])
resources.pop("object_store_memory", None)
print(json.dumps(resources))')"
  export RAY_OVERRIDE_RESOURCES
fi
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=$((32 * 1024 * 1024 * 1024))
export RAY_enable_ray_event=0
export RAY_task_events_report_interval_ms=0
unset RAY_enable_core_worker_ray_event_to_aggregator
unset RAY_DASHBOARD_AGGREGATOR_AGENT_EVENTS_EXPORT_ADDR RAY_DASHBOARD_AGGREGATOR_AGENT_PUBLISHER_MAX_RETRIES

echo "==> experiment $EXPERIMENT in $RUN_DIR"

#-----------------------
# Run: the sibling recipe's settings, apart from the knobs above. The arm's own settings come
# last so they win (DETERMINISTIC's concurrency of 1 over the recipe's 512).
#-----------------------
uv run --isolated --extra "$STRATEGY" "${EXTRAS[@]}" -m "$ENTRYPOINT" \
  data.train_data="['$SUBSET_DIR']" \
  trainer.policy.model.path="$MODEL" \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  trainer.project_name=harbor-compare \
  trainer.run_name="$EXPERIMENT" \
  trainer.logger="$LOGGER" \
  trainer.export_path="$RUN_DIR/exports" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=none \
  harbor_trial_config.trials_dir="$RUN_DIR/trials" \
  harbor_trial_config.environment.type="$SANDBOX" \
  harbor_trial_config.agent.kwargs.temperature="$TEMPERATURE" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_MODEL_LEN" \
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
  trainer.algorithm.temperature=1.0 \
  trainer.policy.optimizer_config.lr="$LR" \
  trainer.strategy="$STRATEGY" \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs.chat_template="$CHAT_TEMPLATE" \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.sampling_params.temperature="$TEMPERATURE" \
  generator.step_wise_trajectories=true \
  generator.apply_overlong_filtering=true \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=5 \
  generator.rate_limit.max_concurrency=512 \
  "${STRATEGY_ARGS[@]}" \
  "${ARM_ARGS[@]}" \
  "$@" 2>&1 | tee "$RUN_DIR/run.log"

echo
echo "==> done: $EXPERIMENT"
if [[ "$GENERATOR" == skycap ]]; then
  echo "    skycap records: $RUN_DIR/skycap"
fi
echo "    Harbor trials:  $RUN_DIR/trials"
echo "    log:            $RUN_DIR/run.log"
