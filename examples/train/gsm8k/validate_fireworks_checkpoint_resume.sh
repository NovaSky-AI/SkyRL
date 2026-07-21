#!/usr/bin/env bash

set -euo pipefail

# Paid end-to-end validation of SkyRL's Fireworks DCP integration:
#   phase A: train a bounded GRPO smoke run, save weights + optimizer state,
#            then tear down;
#   phase B: provision a new trainer/deployment, restore the cross-job DCP,
#            and continue training through the next global step.

SKYRL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${VALIDATION_SUFFIX:=$(date -u +%Y%m%d%H%M%S)-$RANDOM}"
: "${VALIDATION_DIR:=/mnt/local_storage/fireworks-gsm8k-checkpoint-resume-${VALIDATION_SUFFIX}}"
: "${CKPT_PATH:=${VALIDATION_DIR}/checkpoints}"
: "${FIREWORKS_TRAINING_SHAPE:=accounts/fireworks/trainingShapes/qwen3-4b-minimum}"
: "${FIREWORKS_LORA_RANK:=0}"
: "${FIREWORKS_TRAINER_REPLICA_COUNT:=1}"
: "${FIREWORKS_REPLICA_COUNT:=1}"
: "${TRAIN_BATCH_SIZE:=2}"
: "${N_SAMPLES_PER_PROMPT:=4}"
: "${MAX_GENERATE_LENGTH:=1024}"
: "${WANDB_PROJECT:=gsm8k-fireworks}"
: "${SKIP_SAVE_PHASE:=0}"

mkdir -p "${VALIDATION_DIR}/logs" "$CKPT_PATH"

run_phase() {
  local phase=$1
  shift
  local log_path="${VALIDATION_DIR}/logs/${phase}.log"
  (
    cd "$SKYRL_ROOT"
    env \
      FIREWORKS_RUN_CONFIRMED=1 \
      FIREWORKS_TRAINING_SHAPE="$FIREWORKS_TRAINING_SHAPE" \
      FIREWORKS_LORA_RANK="$FIREWORKS_LORA_RANK" \
      FIREWORKS_TRAINER_REPLICA_COUNT="$FIREWORKS_TRAINER_REPLICA_COUNT" \
      FIREWORKS_REPLICA_COUNT="$FIREWORKS_REPLICA_COUNT" \
      TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
      N_SAMPLES_PER_PROMPT="$N_SAMPLES_PER_PROMPT" \
      MAX_GENERATE_LENGTH="$MAX_GENERATE_LENGTH" \
      WANDB_PROJECT="$WANDB_PROJECT" \
      CKPT_INTERVAL=1 \
      CKPT_PATH="$CKPT_PATH" \
      "$@" \
      bash examples/train/gsm8k/run_gsm8k_fireworks_dedicated.sh
  ) 2>&1 | tee -a "$log_path"
}

phase_a_suffix="${VALIDATION_SUFFIX}-save"
if [[ "$SKIP_SAVE_PHASE" != "1" ]]; then
  run_phase save \
    RESOURCE_SUFFIX="$phase_a_suffix" \
    RUN_NAME="gsm8k-fireworks-dcp-save-qwen3-4b-${VALIDATION_SUFFIX}" \
    MAX_TRAINING_STEPS=1 \
    TRAINING_EPOCHS=1 \
    MAX_PAID_RUNTIME_MINUTES=40 \
    RESUME_MODE=null \
    RESUME_PATH=null
fi

latest_file="${CKPT_PATH}/latest_ckpt_global_step.txt"
if [[ ! -s "$latest_file" ]]; then
  printf 'Phase A did not write %s\n' "$latest_file" >&2
  exit 1
fi
saved_step=$(<"$latest_file")
resume_path="${CKPT_PATH}/global_step_${saved_step}"
manifest_path="${resume_path}/policy/fireworks_checkpoint.json"
if [[ ! -s "$manifest_path" ]]; then
  printf 'Phase A did not write %s\n' "$manifest_path" >&2
  exit 1
fi

phase_b_suffix="${VALIDATION_SUFFIX}-resume"
run_phase resume \
  RESOURCE_SUFFIX="$phase_b_suffix" \
  RUN_NAME="gsm8k-fireworks-dcp-resume-qwen3-4b-${VALIDATION_SUFFIX}" \
  MAX_TRAINING_STEPS="$(( saved_step + 1 ))" \
  TRAINING_EPOCHS=2 \
  MAX_PAID_RUNTIME_MINUTES=45 \
  RESUME_MODE=from_path \
  RESUME_PATH="$resume_path"

if ! rg -q "Loaded Fireworks DCP checkpoint:.*optimizer_restored=True" \
  "${VALIDATION_DIR}/logs/resume.log"; then
  printf 'Phase B did not confirm optimizer-state restoration.\n' >&2
  exit 1
fi
if ! rg -q "Successfully loaded complete checkpoint state" \
  "${VALIDATION_DIR}/logs/resume.log"; then
  printf 'Phase B did not confirm SkyRL trainer/dataloader restoration.\n' >&2
  exit 1
fi

printf '%s\n' \
  "Fireworks checkpoint/resume validation passed." \
  "  artifacts: ${VALIDATION_DIR}" \
  "  source checkpoint: ${resume_path}" \
  "  manifest: ${manifest_path}"
