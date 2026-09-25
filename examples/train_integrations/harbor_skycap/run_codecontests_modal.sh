#!/usr/bin/env bash
# A small Harbor training run on CodeContests, sandboxed on Modal, through skycap
# or through the sibling `harbor` integration (the baseline), to compare the two.
#
#   bash examples/train_integrations/harbor_skycap/run_codecontests_modal.sh [extra overrides...]
#
# Does everything: reads the Modal credentials, prepares the tasks under
# /tmp/harbor/data (skipped when they are already there), and trains EPOCHS
# passes over NUM_PROMPTS tasks x GROUP_SIZE samples on this machine's GPUs.
# Each run gets a fresh uuid name, so its trials, skycap records, checkpoints and
# logs all land in their own /tmp/harbor/runs/<name>.
#
#   GENERATOR=skycap|baseline  which integration generates the rollouts (default skycap)
#   TEMPERATURE=0              greedy, so a baseline run and a skycap run can be diffed
#                              token for token with compare_runs.py
#   DETERMINISTIC=1            one trial at a time and no prefix caching, so every request runs
#                              alone and greedy decoding can't flip on batch-dependent numerics
#   TASKS=a,b                  task names to run instead of the first NUM_PROMPTS
#
# Needs: 2 GPUs, and a file holding `MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

#-----------------------
# Knobs
#-----------------------
MODAL_KEY_FILE="${MODAL_KEY_FILE:-$HOME/default/model_key.key}"
GENERATOR="${GENERATOR:-skycap}"
TEMPERATURE="${TEMPERATURE:-1.0}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1.0e-6}"
TASKS="${TASKS:-}"
DETERMINISTIC="${DETERMINISTIC:-0}"
# Non-thinking, so the chat template never strips reasoning from history and a
# rollout stays one path.
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-policy}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"
NUM_GPUS="${NUM_GPUS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
MAX_TURNS="${MAX_TURNS:-8}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-900}"
DATASET="${DATASET:-open-thoughts/CodeContests}"

DATA_ROOT="/tmp/harbor/data"
TASKS_DIR="$DATA_ROOT/$(basename "$DATASET")"

EXPERIMENT="harbor-$GENERATOR-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
RUN_DIR="/tmp/harbor/runs/$EXPERIMENT"
SUBSET_DIR="$RUN_DIR/tasks"

case "$GENERATOR" in
  skycap)
    ENTRYPOINT=examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap
    EXTRAS=(--extra harbor --extra skycap)
    # Each row is already a complete path; merging could fuse two paths.
    ARM_ARGS=(skycap.record_dir="$RUN_DIR/skycap" generator.merge_stepwise_output=false)
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
if [[ "$DETERMINISTIC" == 1 ]]; then
  ARM_ARGS+=(
    generator.rate_limit.enabled=true
    generator.rate_limit.max_concurrency=1
    generator.inference_engine.engine_init_kwargs.enable_prefix_caching=false
  )
fi

#-----------------------
# Modal credentials
#-----------------------
if [[ ! -f "$MODAL_KEY_FILE" ]]; then
  echo "no Modal credentials at $MODAL_KEY_FILE (set MODAL_KEY_FILE)" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$MODAL_KEY_FILE"
set +a
: "${MODAL_TOKEN_ID:?$MODAL_KEY_FILE must set MODAL_TOKEN_ID}"
: "${MODAL_TOKEN_SECRET:?$MODAL_KEY_FILE must set MODAL_TOKEN_SECRET}"

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
echo "==> $NUM_PROMPTS tasks in $SUBSET_DIR"

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
# Run
#-----------------------
uv run --isolated --extra fsdp "${EXTRAS[@]}" -m "$ENTRYPOINT" \
  "${ARM_ARGS[@]}" \
  data.train_data="['$SUBSET_DIR']" \
  trainer.policy.model.path="$MODEL" \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  trainer.project_name=harbor-compare \
  trainer.run_name="$EXPERIMENT" \
  trainer.logger=console \
  trainer.export_path="$RUN_DIR/exports" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=none \
  harbor_trial_config.trials_dir="$RUN_DIR/trials" \
  harbor_trial_config.environment.type=modal \
  harbor_trial_config.agent.override_timeout_sec="$AGENT_TIMEOUT_SEC" \
  harbor_trial_config.agent.kwargs.max_turns="$MAX_TURNS" \
  harbor_trial_config.agent.kwargs.temperature="$TEMPERATURE" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_GENERATE_LENGTH" \
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
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.algorithm.temperature=1.0 \
  trainer.policy.optimizer_config.lr="$LR" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.sampling_params.temperature="$TEMPERATURE" \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.step_wise_trajectories=true \
  generator.batched=false \
  "$@" 2>&1 | tee "$RUN_DIR/run.log"

echo
echo "==> done: $EXPERIMENT"
if [[ "$GENERATOR" == skycap ]]; then
  echo "    skycap records: $RUN_DIR/skycap"
fi
echo "    Harbor trials:  $RUN_DIR/trials"
echo "    log:            $RUN_DIR/run.log"
