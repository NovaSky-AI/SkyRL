#!/bin/bash
set -x

# =============================================================================
# Tinker RL Training for Math Reasoning Task
# =============================================================================
# This script demonstrates how to train a model on math reasoning using:
# - GRPO (Group Relative Policy Optimization) for advantages
# - PPO loss for stable training
# =============================================================================

# Data paths
DATA_DIR="${DATA_DIR:-/home/ec2-user/data/nq_search}"
DATASET_FILE="${DATASET_FILE:-${DATA_DIR}/train.parquet}"
EVAL_DATASET_FILE="${EVAL_DATASET_FILE:-${DATA_DIR}/test.parquet}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./tinker_outputs/search-r1}"
mkdir -p "$OUTPUT_DIR"

# Model configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
LORA_RANK="${LORA_RANK:-32}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_EVERY="${SAVE_EVERY:-2}"
EVAL_EVERY="${EVAL_EVERY:-10}"

# RL configuration
LOSS_FN="${LOSS_FN:-ppo}"
GROUP_SIZE="${GROUP_SIZE:-8}"  # Should match num_trajectories in YAML
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-false}"

# Logging
WANDB_PROJECT="${WANDB_PROJECT:-tinker-reasoning}"
WANDB_NAME="${WANDB_NAME:-tinker-qwen3-4b-search-r1-grpo}"
RESUME_EXP_NAME="${RESUME_EXP_NAME:-tinker-qwen3-4b-search-r1-grpo_1030T19:52:09}"

# Task configuration
TASK_YAML="./examples/run_tinker/tinker_reasoning_search_r1.yaml"

echo "================================================"
echo "Tinker RL Training Configuration"
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
    eval_batch_size="$EVAL_BATCH_SIZE" \
    learning_rate="$LEARNING_RATE" \
    lora_rank="$LORA_RANK" \
    max_steps="$MAX_STEPS" \
    save_every="$SAVE_EVERY" \
    eval_every="$EVAL_EVERY" \
    loss_fn="$LOSS_FN" \
    group_size="$GROUP_SIZE" \
    normalize_advantages="$NORMALIZE_ADVANTAGES" \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME" \
    resume_exp_name="$RESUME_EXP_NAME" \
    log_dir="$OUTPUT_DIR" \
    "$@"

echo "================================================"
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}/tinker_output/${WANDB_NAME}_*"
echo "================================================"

