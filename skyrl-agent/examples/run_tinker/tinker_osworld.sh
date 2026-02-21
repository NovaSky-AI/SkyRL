#!/bin/bash
# set -x

# =============================================================================
# Tinker RL Training for MemAgent Task
# =============================================================================
# This script demonstrates how to train a model on ruler/hotpotqa using:
# - GRPO (Group Relative Policy Optimization) for advantages
# - PPO loss for stable training
# - MemAgent tool with multi-turn interactions
# =============================================================================

# Data paths
DATASET_FILE="/home/ubuntu/shuo/osworld/OSWorld_llm_agentsynth/osworld_train_8.parquet"

EVAL_DATASET_FILE="/home/ubuntu/shuo/osworld/OSWorld_llm_agentsynth/osworld_train_8.parquet"

# Output directory
NAME="${NAME:-jan03_qwen3_8b_osworld_tinker_lr4e_5_rank128}"
OUTPUT_DIR="/home/ubuntu/shuo/osworld/checkpoints/${NAME}"
mkdir -p "$OUTPUT_DIR"

# Model configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
LORA_RANK="${LORA_RANK:-128}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
MAX_STEPS="${MAX_STEPS:-50}"
SAVE_EVERY="${SAVE_EVERY:-5}"
EVAL_EVERY="${EVAL_EVERY:-10}"

# RL configuration
LOSS_FN="${LOSS_FN:-ppo}"
GROUP_SIZE="${GROUP_SIZE:-8}"  # Should match num_trajectories in YAML
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-false}"

# Logging
WANDB_PROJECT="${WANDB_PROJECT:-tinker-osw}"
WANDB_NAME="${WANDB_NAME:-${NAME}}"

# Task configuration
TASK_YAML="./examples/run_tinker/tinker_osworld.yaml"

echo "================================================"
echo "Tinker RL Training Configuration - OSWorld"
echo "================================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_FILE"
echo "Task YAML: $TASK_YAML"
echo "Batch Size: $BATCH_SIZE"
echo "Group Size (GRPO): $GROUP_SIZE"
echo "Max Steps: $MAX_STEPS"
echo "Output: $OUTPUT_DIR"
echo "================================================"

# Run training
# UV_NO_SYNC=1 prevents uv from trying to (re)install dependencies (like vllm);
# make sure required deps are already installed in the active env.
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 UV_NO_SYNC=1 uv run --active --extra tinker --env-file .env -m skyrl_agent.integrations.tinker.tinker_train \
    model_name="$MODEL_NAME" \
    skyrl_agent_task_yaml="$TASK_YAML" \
    dataset_file="$DATASET_FILE" \
    eval_dataset_file="$EVAL_DATASET_FILE" \
    batch_size="$BATCH_SIZE" \
    learning_rate="$LEARNING_RATE" \
    lora_rank="$LORA_RANK" \
    max_steps="$MAX_STEPS" \
    save_every="$SAVE_EVERY" \
    loss_fn="$LOSS_FN" \
    group_size="$GROUP_SIZE" \
    normalize_advantages="$NORMALIZE_ADVANTAGES" \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME" \
    log_dir="$OUTPUT_DIR" \
    "$@"

echo "================================================"
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/${WANDB_NAME}_*"
echo "================================================"
