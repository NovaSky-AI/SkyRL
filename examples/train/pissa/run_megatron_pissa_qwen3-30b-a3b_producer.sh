#!/usr/bin/env bash

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-30B-A3B}"
LORA_RANK="${LORA_RANK:-64}"
PISSA_INIT_DIR="${PISSA_INIT_DIR:-$HOME/pissa/qwen3_30b_a3b_r${LORA_RANK}}"
DEFAULT_CONFIG_OVERRIDES='{"trainer.placement.policy_num_gpus_per_node":8,"trainer.policy.megatron_config.tensor_model_parallel_size":4,"trainer.policy.megatron_config.pipeline_model_parallel_size":1,"trainer.policy.megatron_config.context_parallel_size":1,"trainer.policy.megatron_config.expert_model_parallel_size":8,"trainer.policy.megatron_config.expert_tensor_parallel_size":1}'
CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-$DEFAULT_CONFIG_OVERRIDES}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.pissa_init \
  --base-model "$BASE_MODEL" \
  --rank "$LORA_RANK" \
  --output-dir "$PISSA_INIT_DIR" \
  --config-overrides "$CONFIG_OVERRIDES"
