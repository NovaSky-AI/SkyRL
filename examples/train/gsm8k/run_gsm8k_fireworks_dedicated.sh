#!/usr/bin/env bash

set -euo pipefail
set -x

# Bounded, policy-only GRPO validation on a dedicated trainer and linked
# dedicated rollout deployment. Both uniquely named resources are deleted
# on normal completion, failure, interruption, or the wall-clock timeout.

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${FIREWORKS_BASE_MODEL:=accounts/fireworks/models/qwen3-4b}"
: "${FIREWORKS_TOKENIZER_MODEL:=Qwen/Qwen3-4B}"
: "${FIREWORKS_TRAINING_SHAPE:=accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora}"
: "${FIREWORKS_MAX_SEQ_LEN:=32768}"
: "${FIREWORKS_LORA_RANK:=8}"
: "${FIREWORKS_TRAINER_REPLICA_COUNT:=1}"
: "${FIREWORKS_REPLICA_COUNT:=1}"
: "${FIREWORKS_TRAINER_CHIPS_PER_REPLICA:=1}"
: "${FIREWORKS_ROLLOUT_CHIPS_PER_REPLICA:=1}"
: "${TRAIN_BATCH_SIZE:=2}"
: "${N_SAMPLES_PER_PROMPT:=4}"
: "${MAX_GENERATE_LENGTH:=1024}"
: "${MAX_TRAINING_STEPS:=1}"
: "${TRAINING_EPOCHS:=$MAX_TRAINING_STEPS}"
: "${MAX_PAID_RUNTIME_MINUTES:=20}"
: "${LOGGER:=wandb}"
: "${WANDB_PROJECT:=gsm8k-fireworks}"
: "${CKPT_INTERVAL:=-1}"
: "${CKPT_PATH:=$HOME/ckpts}"
: "${RESUME_MODE:=null}"
: "${RESUME_PATH:=null}"
: "${RESOURCE_SUFFIX:=$(date -u +%Y%m%d%H%M%S)-$RANDOM}"
: "${FIREWORKS_TRAINER_JOB_ID:=skyrl-smoke-gsm8k-${RESOURCE_SUFFIX}-trainer}"
: "${FIREWORKS_DEPLOYMENT_ID:=skyrl-smoke-gsm8k-${RESOURCE_SUFFIX}-rollout}"
: "${RUN_NAME:=gsm8k-fireworks-dedicated-qwen3-4b-${RESOURCE_SUFFIX}}"

for resource_id in "$FIREWORKS_TRAINER_JOB_ID" "$FIREWORKS_DEPLOYMENT_ID"; do
  if [[ "$resource_id" != skyrl-smoke-* || "$resource_id" == */* ]]; then
    printf 'Refusing unsafe smoke resource ID: %s\n' "$resource_id" >&2
    exit 2
  fi
done

for positive_integer in \
  "$FIREWORKS_TRAINER_REPLICA_COUNT" \
  "$FIREWORKS_REPLICA_COUNT" \
  "$FIREWORKS_TRAINER_CHIPS_PER_REPLICA" \
  "$FIREWORKS_ROLLOUT_CHIPS_PER_REPLICA"; do
  if [[ ! "$positive_integer" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Expected a positive integer resource count, got: %s\n' "$positive_integer" >&2
    exit 2
  fi
done

FIREWORKS_TRAINER_CHIP_COUNT=$((
  FIREWORKS_TRAINER_REPLICA_COUNT * FIREWORKS_TRAINER_CHIPS_PER_REPLICA
))
FIREWORKS_ROLLOUT_CHIP_COUNT=$((
  FIREWORKS_REPLICA_COUNT * FIREWORKS_ROLLOUT_CHIPS_PER_REPLICA
))
FIREWORKS_TOTAL_CHIP_COUNT=$((
  FIREWORKS_TRAINER_CHIP_COUNT + FIREWORKS_ROLLOUT_CHIP_COUNT
))
FIREWORKS_COMBINED_RATE_USD_PER_HOUR=$((FIREWORKS_TOTAL_CHIP_COUNT * 10))
FIREWORKS_COST_CEILING_CENTS=$((
  (FIREWORKS_COMBINED_RATE_USD_PER_HOUR * MAX_PAID_RUNTIME_MINUTES * 100 + 59) / 60
))

if [[ "${FIREWORKS_RUN_CONFIRMED:-0}" != "1" ]]; then
  set +x
  printf '%s\n' \
    "This command creates real paid Fireworks dedicated resources." \
    "Resolved smoke-test plan:" \
    "  base model: ${FIREWORKS_BASE_MODEL}" \
    "  tokenizer: ${FIREWORKS_TOKENIZER_MODEL}" \
    "  training shape: ${FIREWORKS_TRAINING_SHAPE}" \
    "  trainer job ID: ${FIREWORKS_TRAINER_JOB_ID}" \
    "  deployment ID: ${FIREWORKS_DEPLOYMENT_ID}" \
    "  trainer: ${FIREWORKS_TRAINER_REPLICA_COUNT} data-parallel replica(s) x ${FIREWORKS_TRAINER_CHIPS_PER_REPLICA} B200 = ${FIREWORKS_TRAINER_CHIP_COUNT} B200(s)" \
    "  rollout: ${FIREWORKS_REPLICA_COUNT} replica(s) x ${FIREWORKS_ROLLOUT_CHIPS_PER_REPLICA} B200 = ${FIREWORKS_ROLLOUT_CHIP_COUNT} B200(s)" \
    "  public combined rate while all are active: approximately USD ${FIREWORKS_COMBINED_RATE_USD_PER_HOUR}/hour" \
    "  wall-clock cap: ${MAX_PAID_RUNTIME_MINUTES} minutes (approximately USD $(( FIREWORKS_COST_CEILING_CENTS / 100 )).$(printf '%02d' $(( FIREWORKS_COST_CEILING_CENTS % 100 ))) ceiling at that rate)" \
    "  optimizer steps: ${MAX_TRAINING_STEPS} GRPO step(s)" \
    "  prompt groups: ${TRAIN_BATCH_SIZE}" \
    "  completions per prompt: ${N_SAMPLES_PER_PROMPT}" \
    "  maximum generated tokens: $(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT * MAX_GENERATE_LENGTH ))" \
    "  checkpoint interval: ${CKPT_INTERVAL} (path: ${CKPT_PATH})" \
    "  resume: ${RESUME_MODE} (path: ${RESUME_PATH})" \
    "  HF export: disabled" \
    "  cleanup: delete the new deployment and stop/delete the new trainer, then audit both" \
    "Rerun with FIREWORKS_RUN_CONFIRMED=1 after approving this exact plan."
  exit 2
fi

set +x
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

timeout --signal=INT --kill-after=2m "${MAX_PAID_RUNTIME_MINUTES}m" \
  uv run --isolated --extra fireworks -m skyrl.train.entrypoints.main_fireworks \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="[]" \
  trainer.strategy=fireworks \
  trainer.fireworks.base_model="$FIREWORKS_BASE_MODEL" \
  trainer.fireworks.max_seq_len="$FIREWORKS_MAX_SEQ_LEN" \
  trainer.fireworks.training_shape_id="$FIREWORKS_TRAINING_SHAPE" \
  trainer.fireworks.trainer_job_id="$FIREWORKS_TRAINER_JOB_ID" \
  trainer.fireworks.trainer_replica_count="$FIREWORKS_TRAINER_REPLICA_COUNT" \
  trainer.fireworks.deployment_id="$FIREWORKS_DEPLOYMENT_ID" \
  trainer.fireworks.replica_count="$FIREWORKS_REPLICA_COUNT" \
  trainer.fireworks.trainer_timeout_s=900 \
  trainer.fireworks.deployment_timeout_s=900 \
  trainer.fireworks.hotload_timeout_s=600 \
  trainer.fireworks.request_timeout_s=600 \
  trainer.fireworks.sampling_timeout_s=300 \
  trainer.fireworks.cleanup_on_exit=true \
  trainer.fireworks.cleanup_deployment_on_close=delete \
  trainer.fireworks.snapshot_prefix="$FIREWORKS_TRAINER_JOB_ID" \
  trainer.policy.model.path="$FIREWORKS_TOKENIZER_MODEL" \
  trainer.policy.model.lora.rank="$FIREWORKS_LORA_RANK" \
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
  trainer.epochs="$TRAINING_EPOCHS" \
  trainer.max_training_steps="$MAX_TRAINING_STEPS" \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))" \
  trainer.micro_train_batch_size_per_gpu="$(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))" \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=512 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.resume_path="$RESUME_PATH" \
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
  trainer.project_name="$WANDB_PROJECT" \
  trainer.run_name="$RUN_NAME" \
  "$@"
