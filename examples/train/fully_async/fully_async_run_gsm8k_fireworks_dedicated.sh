#!/usr/bin/env bash

set -euo pipefail

# Long-lived, fully asynchronous, full-parameter Qwen3-4B GRPO on Fireworks.
#
# The training shape owns one B200 per trainer replica. One trainer replica and
# one one-B200 rollout replica are created by default, for two B200s total.
# There is deliberately no wall-clock timeout and no max_training_steps
# override; interrupting this script runs exact-ID cleanup.

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${FIREWORKS_BASE_MODEL:=accounts/fireworks/models/qwen3-4b}"
: "${FIREWORKS_TOKENIZER_MODEL:=Qwen/Qwen3-4B}"
: "${FIREWORKS_TRAINING_SHAPE:=accounts/fireworks/trainingShapes/qwen3-4b-minimum}"
: "${FIREWORKS_MAX_SEQ_LEN:=32768}"
: "${FIREWORKS_TRAINER_REPLICA_COUNT:=1}"
: "${FIREWORKS_REPLICA_COUNT:=1}"
: "${TRAIN_BATCH_SIZE:=64}"
: "${N_SAMPLES_PER_PROMPT:=5}"
: "${MAX_STALENESS_STEPS:=2}"
: "${NUM_PARALLEL_GENERATION_WORKERS:=$(( TRAIN_BATCH_SIZE * (MAX_STALENESS_STEPS + 1) ))}"
: "${MAX_GENERATE_LENGTH:=1024}"
: "${TRAINING_EPOCHS:=1000}"
: "${LOGGER:=wandb}"
: "${RESOURCE_SUFFIX:=$(date -u +%Y%m%d%H%M%S)-$RANDOM}"
: "${FIREWORKS_TRAINER_JOB_ID:=skyrl-smoke-gsm8k-async-${RESOURCE_SUFFIX}-trainer}"
: "${FIREWORKS_DEPLOYMENT_ID:=skyrl-smoke-gsm8k-async-${RESOURCE_SUFFIX}-rollout}"
: "${RUN_NAME:=gsm8k-fireworks-fully-async-qwen3-4b-b64n5-s2-1train2rollout-${RESOURCE_SUFFIX}}"
: "${LOG_DIR:=/tmp/skyrl-fireworks-fully-async}"
: "${DRIVER_LOG:=${LOG_DIR}/${RUN_NAME}.log}"

for resource_id in "$FIREWORKS_TRAINER_JOB_ID" "$FIREWORKS_DEPLOYMENT_ID"; do
  if [[ "$resource_id" != skyrl-smoke-* || "$resource_id" == */* ]]; then
    printf 'Refusing unsafe Fireworks resource ID: %s\n' "$resource_id" >&2
    exit 2
  fi
done

for positive_integer in \
  "$FIREWORKS_TRAINER_REPLICA_COUNT" \
  "$FIREWORKS_REPLICA_COUNT" \
  "$TRAIN_BATCH_SIZE" \
  "$N_SAMPLES_PER_PROMPT" \
  "$NUM_PARALLEL_GENERATION_WORKERS" \
  "$TRAINING_EPOCHS"; do
  if [[ ! "$positive_integer" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Expected a positive integer, got: %s\n' "$positive_integer" >&2
    exit 2
  fi
done

expected_workers=$(( TRAIN_BATCH_SIZE * (MAX_STALENESS_STEPS + 1) ))
if [[ "$NUM_PARALLEL_GENERATION_WORKERS" -ne "$expected_workers" ]]; then
  printf 'Expected num_parallel_generation_workers=(max_staleness_steps+1)*batch=%s, got %s\n' \
    "$expected_workers" "$NUM_PARALLEL_GENERATION_WORKERS" >&2
  exit 2
fi

FIREWORKS_TOTAL_CHIP_COUNT=$(( FIREWORKS_TRAINER_REPLICA_COUNT + FIREWORKS_REPLICA_COUNT ))
FIREWORKS_COMBINED_RATE_USD_PER_HOUR=$(( FIREWORKS_TOTAL_CHIP_COUNT * 10 ))

mkdir -p "$LOG_DIR"
exec > >(tee -a "$DRIVER_LOG") 2>&1

if [[ "${FIREWORKS_RUN_CONFIRMED:-0}" != "1" ]]; then
  printf '%s\n' \
    "This command creates real paid Fireworks dedicated resources." \
    "Resolved fully-async plan:" \
    "  full-parameter model: ${FIREWORKS_BASE_MODEL}" \
    "  training shape: ${FIREWORKS_TRAINING_SHAPE} (one B200 per replica)" \
    "  trainer: ${FIREWORKS_TRAINER_REPLICA_COUNT} data-parallel one-B200 replica(s)" \
    "  rollout: ${FIREWORKS_REPLICA_COUNT} one-B200 replica(s)" \
    "  total: ${FIREWORKS_TOTAL_CHIP_COUNT} B200s, approximately USD ${FIREWORKS_COMBINED_RATE_USD_PER_HOUR}/hour at the public rate" \
    "  batch: ${TRAIN_BATCH_SIZE} prompts x ${N_SAMPLES_PER_PROMPT} completions" \
    "  staleness: ${MAX_STALENESS_STEPS}; generation workers: ${NUM_PARALLEL_GENERATION_WORKERS}" \
    "  duration: no wall-clock timeout; ${TRAINING_EPOCHS} epochs is the natural upper bound" \
    "  trainer ID: ${FIREWORKS_TRAINER_JOB_ID}" \
    "  deployment ID: ${FIREWORKS_DEPLOYMENT_ID}" \
    "  driver log: ${DRIVER_LOG}" \
    "  cleanup: delete the exact deployment and stop/delete the exact trainer on exit" \
    "Rerun with FIREWORKS_RUN_CONFIRMED=1 after approving this exact plan."
  exit 2
fi

if [[ -z "${FIREWORKS_API_KEY:-}" ]]; then
  printf 'FIREWORKS_API_KEY is not set in this shell.\n' >&2
  exit 2
fi
if [[ "$LOGGER" == "wandb" && -z "${WANDB_API_KEY:-}" ]]; then
  printf 'WANDB_API_KEY is not set in this shell (or set LOGGER=console).\n' >&2
  exit 2
fi

cleanup_resources() {
  local command_status=$?
  trap - EXIT INT TERM
  set +e
  uv run --isolated --extra fireworks examples/train/gsm8k/fireworks_dedicated_cleanup.py \
    --trainer-job-id "$FIREWORKS_TRAINER_JOB_ID" \
    --deployment-id "$FIREWORKS_DEPLOYMENT_ID"
  local cleanup_status=$?
  if [[ "$cleanup_status" -ne 0 ]]; then
    printf 'ERROR: secondary Fireworks resource cleanup failed.\n' >&2
    exit "$cleanup_status"
  fi
  exit "$command_status"
}
trap cleanup_resources EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set -x
uv run --isolated --extra fireworks -m skyrl.train.entrypoints.main_fireworks \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="[]" \
  trainer.strategy=fireworks \
  trainer.fireworks.infrastructure=dedicated \
  trainer.fireworks.base_model="$FIREWORKS_BASE_MODEL" \
  trainer.fireworks.max_seq_len="$FIREWORKS_MAX_SEQ_LEN" \
  trainer.fireworks.training_shape_id="$FIREWORKS_TRAINING_SHAPE" \
  trainer.fireworks.trainer_job_id="$FIREWORKS_TRAINER_JOB_ID" \
  trainer.fireworks.trainer_replica_count="$FIREWORKS_TRAINER_REPLICA_COUNT" \
  trainer.fireworks.deployment_id="$FIREWORKS_DEPLOYMENT_ID" \
  trainer.fireworks.replica_count="$FIREWORKS_REPLICA_COUNT" \
  trainer.fireworks.trainer_timeout_s=900 \
  trainer.fireworks.deployment_timeout_s=900 \
  trainer.fireworks.hotload_timeout_s=900 \
  trainer.fireworks.request_timeout_s=1800 \
  trainer.fireworks.sampling_timeout_s=1800 \
  trainer.fireworks.cleanup_on_exit=true \
  trainer.fireworks.cleanup_deployment_on_close=delete \
  trainer.fireworks.snapshot_prefix="$FIREWORKS_TRAINER_JOB_ID" \
  trainer.policy.model.path="$FIREWORKS_TOKENIZER_MODEL" \
  trainer.policy.model.lora.rank=0 \
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.policy.optimizer_config.adam_betas="[0.9,0.95]" \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.scheduler=constant_with_warmup \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=rollout_is \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.zero_variance_filter=true \
  trainer.critic.model.path=null \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=false \
  trainer.fully_async.enabled=true \
  trainer.fully_async.max_staleness_steps="$MAX_STALENESS_STEPS" \
  trainer.fully_async.num_parallel_generation_workers="$NUM_PARALLEL_GENERATION_WORKERS" \
  trainer.fully_async.sample_full_batch=false \
  trainer.fully_async.clear_kv_cache_on_weight_sync=false \
  trainer.epochs="$TRAINING_EPOCHS" \
  trainer.max_training_steps=null \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))" \
  trainer.micro_train_batch_size_per_gpu="$(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))" \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=512 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=null \
  trainer.enable_ray_gpu_monitor=false \
  generator.inference_engine.backend=fireworks \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.enable_ray_prometheus_stats=false \
  generator.batched=false \
  generator.max_turns=1 \
  generator.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT" \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.logprobs=1 \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name=gsm8k-fireworks \
  trainer.run_name="$RUN_NAME" \
  "$@"
