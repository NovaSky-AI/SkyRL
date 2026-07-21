#!/usr/bin/env bash

set -euo pipefail

# A paid, one-step validation of Fireworks data-parallel trainer scaling.
# qwen3-4b-minimum is one node with one B200 per trainer replica, so setting
# trainer_replica_count=2 requests two logical one-B200 trainer replicas. The
# rollout deployment remains one B200. The delegated launcher prints the
# resolved three-B200 cost ceiling and requires FIREWORKS_RUN_CONFIRMED=1.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export FIREWORKS_TRAINING_SHAPE="${FIREWORKS_TRAINING_SHAPE:-accounts/fireworks/trainingShapes/qwen3-4b-minimum}"
export FIREWORKS_LORA_RANK="${FIREWORKS_LORA_RANK:-0}"
export FIREWORKS_TRAINER_REPLICA_COUNT="${FIREWORKS_TRAINER_REPLICA_COUNT:-2}"
export FIREWORKS_TRAINER_CHIPS_PER_REPLICA="${FIREWORKS_TRAINER_CHIPS_PER_REPLICA:-1}"
export FIREWORKS_REPLICA_COUNT="${FIREWORKS_REPLICA_COUNT:-1}"
export FIREWORKS_ROLLOUT_CHIPS_PER_REPLICA="${FIREWORKS_ROLLOUT_CHIPS_PER_REPLICA:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-5}"
export MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-1}"
export TRAINING_EPOCHS="${TRAINING_EPOCHS:-1}"
export MAX_PAID_RUNTIME_MINUTES="${MAX_PAID_RUNTIME_MINUTES:-30}"
export RUN_NAME="${RUN_NAME:-gsm8k-fireworks-qwen3-4b-2x-b200-trainer-replicas}"

exec bash "$SCRIPT_DIR/run_gsm8k_fireworks_dedicated.sh"
