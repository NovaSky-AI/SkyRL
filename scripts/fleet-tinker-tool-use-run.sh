#!/usr/bin/env bash
# Launch Fleet tool-use training via Tinker hosted service.
# Mirrors fleet-35b-run.sh config but uses the Tinker backend.
#
# Required env vars: TINKER_API_KEY, FLEET_API_KEY, WANDB_API_KEY
# Optional: TINKER_API_URL (SDK uses default if not set)
set -euo pipefail

export TINKER_API_KEY="${TINKER_API_KEY:?Set TINKER_API_KEY}"
export TINKER_API_URL="${TINKER_API_URL:-}"
export FLEET_API_KEY="${FLEET_API_KEY:?Set FLEET_API_KEY}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"

cd "$(dirname "$0")/.."  # cd to SkyRL-v2 root

python -m integrations.fleet.entrypoints.main_fleet_tinker \
    --model-name Qwen/Qwen3.5-35B-A3B \
    --tasks-file "${TASKS_FILE:?Set TASKS_FILE}" \
    --dataset-file "${DATASET_FILE:?Set DATASET_FILE}" \
    --batch-size 16 \
    --learning-rate 5.0e-7 \
    --lora-rank 16 \
    --max-steps 200 \
    --max-turns 50 \
    --max-generate-length 4096 \
    --max-input-length 96000 \
    --n-samples-per-prompt 8 \
    --eval-every 20 \
    --temperature 0.9 \
    --top-p 0.95 \
    --stop-sequences '["</tool_call>"]' \
    --loss-fn ppo \
    --wandb-project fleet-tinker-grpo \
    "$@"
