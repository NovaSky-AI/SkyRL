#!/usr/bin/env bash

set -euo pipefail

: "${FIREWORKS_BASE_MODEL:=accounts/fireworks/models/qwen3-4b}"
: "${FIREWORKS_TOKENIZER_MODEL:=Qwen/Qwen3-4B}"
: "${FIREWORKS_TRAINING_SHAPE:?Set FIREWORKS_TRAINING_SHAPE to a validated dedicated training shape}"
: "${FIREWORKS_MAX_SEQ_LEN:=32768}"
: "${FIREWORKS_LORA_RANK:=8}"
: "${TRAIN_DATASET:=yahma/alpaca-cleaned}"
: "${TRAIN_SPLIT:=train[:100]}"
: "${MAX_LENGTH:=512}"
: "${BATCH_SIZE:=4}"
: "${NUM_STEPS:=2}"
: "${MAX_PAID_RUNTIME_MINUTES:=15}"
: "${RESOURCE_SUFFIX:=$(date -u +%Y%m%d%H%M%S)-$RANDOM}"
: "${FIREWORKS_TRAINER_JOB_ID:=skyrl-smoke-sft-${RESOURCE_SUFFIX}-trainer}"
: "${RUN_NAME:=skyrl-sft-fireworks-${RESOURCE_SUFFIX}}"
: "${LOG_FILE:=/tmp/skyrl-sft-fireworks-${RESOURCE_SUFFIX}.log}"

if [[ "$FIREWORKS_TRAINER_JOB_ID" != skyrl-smoke-* || "$FIREWORKS_TRAINER_JOB_ID" == */* ]]; then
  printf 'Refusing unsafe smoke trainer ID: %s\n' "$FIREWORKS_TRAINER_JOB_ID" >&2
  exit 2
fi

if [[ "${FIREWORKS_RUN_CONFIRMED:-0}" != "1" ]]; then
  printf '%s\n' \
    "This command creates a paid Fireworks dedicated trainer." \
    "Resolved smoke-test plan:" \
    "  base model: ${FIREWORKS_BASE_MODEL}" \
    "  tokenizer: ${FIREWORKS_TOKENIZER_MODEL}" \
    "  training shape: ${FIREWORKS_TRAINING_SHAPE}" \
    "  trainer job ID: ${FIREWORKS_TRAINER_JOB_ID}" \
    "  rollout deployment: none" \
    "  dataset: ${TRAIN_DATASET} (${TRAIN_SPLIT})" \
    "  optimizer steps: ${NUM_STEPS}" \
    "  batch size: ${BATCH_SIZE}" \
    "  wall-clock cap: ${MAX_PAID_RUNTIME_MINUTES} minutes" \
    "  pricing: https://fireworks.ai/pricing" \
    "  log: ${LOG_FILE}" \
    "  cleanup: delete and audit the exact trainer ID" \
    "Rerun with FIREWORKS_RUN_CONFIRMED=1 after approving this exact plan."
  exit 2
fi

if [[ -z "${FIREWORKS_API_KEY:-}" ]]; then
  printf 'FIREWORKS_API_KEY is not set.\n' >&2
  exit 2
fi

cleanup_resources() {
  local command_status=$?
  trap - EXIT INT TERM
  set +e
  uv run --isolated --extra fireworks examples/train/gsm8k/fireworks_dedicated_cleanup.py \
    --trainer-job-id "$FIREWORKS_TRAINER_JOB_ID" >> "$LOG_FILE" 2>&1
  local cleanup_status=$?
  if [[ "$cleanup_status" -ne 0 ]]; then
    printf 'ERROR: secondary Fireworks trainer cleanup failed; inspect %s.\n' "$LOG_FILE" >&2
    exit "$cleanup_status"
  fi
  exit "$command_status"
}
trap cleanup_resources EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

timeout --signal=INT --kill-after=2m "${MAX_PAID_RUNTIME_MINUTES}m" \
  uv run --isolated --extra fireworks -m skyrl.train.entrypoints.main_fireworks_sft \
  strategy=fireworks \
  model.path="$FIREWORKS_TOKENIZER_MODEL" \
  model.lora.rank="$FIREWORKS_LORA_RANK" \
  fireworks.base_model="$FIREWORKS_BASE_MODEL" \
  fireworks.max_seq_len="$FIREWORKS_MAX_SEQ_LEN" \
  fireworks.training_shape_id="$FIREWORKS_TRAINING_SHAPE" \
  fireworks.trainer_job_id="$FIREWORKS_TRAINER_JOB_ID" \
  fireworks.trainer_replica_count=1 \
  fireworks.request_timeout_s=600 \
  fireworks.cleanup_on_exit=true \
  train_datasets="['$TRAIN_DATASET']" \
  train_dataset_splits="['$TRAIN_SPLIT']" \
  max_length="$MAX_LENGTH" \
  batch_size="$BATCH_SIZE" \
  num_steps="$NUM_STEPS" \
  remove_microbatch_padding=false \
  use_sequence_packing=false \
  enable_ray_gpu_monitor=false \
  optimizer_config.lr=1.0e-5 \
  optimizer_config.adam_betas="[0.9,0.95]" \
  optimizer_config.weight_decay=0.0 \
  optimizer_config.max_grad_norm=1.0 \
  optimizer_config.num_warmup_steps=0 \
  optimizer_config.scheduler=constant_with_warmup \
  logger=console \
  project_name=skyrl_sft \
  run_name="$RUN_NAME" \
  ckpt_path="" \
  ckpt_interval=0 \
  hf_save_interval=0 \
  resume_from="" \
  "$@" > "$LOG_FILE" 2>&1

printf 'Fireworks SFT smoke run completed. Log: %s\n' "$LOG_FILE"
