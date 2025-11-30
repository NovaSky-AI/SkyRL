#!/bin/bash
set -x

# =============================================================================
# Tinker RL Training for MemAgent Task
# =============================================================================
# This script demonstrates how to train a model on ruler/hotpotqa using:
# - GRPO (Group Relative Policy Optimization) for advantages
# - PPO loss for stable training
# - MemAgent tool with multi-turn interactions
# =============================================================================

# Data paths
DATASET_FILE="/mnt/shared_storage/dacheng/ruler_hotpotqa_train_32k_skyagent.parquet"

EVAL_DATASET_FILE="/mnt/shared_storage/dacheng/ruler_hotpotqa_eval_800_skyagent.parquet"

# Output directory
NAME="${NAME:-nov24_qwen3_8b_ruler_max8k_external_min0turns_max30turns_memagent_tinker_lr4e_5_rank128}"
OUTPUT_DIR="/mnt/local_storage/checkpoints/${NAME}"
mkdir -p "$OUTPUT_DIR"

# Model configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
LORA_RANK="${LORA_RANK:-128}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
MAX_STEPS="${MAX_STEPS:-12}"
SAVE_EVERY="${SAVE_EVERY:-5}"
EVAL_EVERY="${EVAL_EVERY:-10}"

# RL configuration
LOSS_FN="${LOSS_FN:-ppo}"
GROUP_SIZE="${GROUP_SIZE:-8}"  # Should match num_trajectories in YAML
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-false}"

# Logging
WANDB_PROJECT="${WANDB_PROJECT:-tinker-reasoning}"
WANDB_NAME="${WANDB_NAME:-${NAME}}"

# Task configuration
TASK_YAML="./examples/run_tinker/tinker_memagent.yaml"

echo "================================================"
echo "Tinker RL Training Configuration - MemAgent"
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
uv run --isolated --extra tinker --env-file .env -m skyrl_agent.integrations.tinker.tinker_train \
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
