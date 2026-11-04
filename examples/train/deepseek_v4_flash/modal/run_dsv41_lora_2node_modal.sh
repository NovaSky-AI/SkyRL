#!/bin/bash
# DeepSeek-V4.1-Flash LoRA DAPO (examples/train/deepseek_v4_flash/run_dapo_dsv41_flash_lora_sync_2node.sh) on a
# 2 x B200:8 Modal cluster:
#   MODAL_NODES=2 MODAL_GPU=B200:8 modal run examples/train_integrations/modal/main_multinode.py \
#     --command "WANDB_API_KEY=... bash examples/train/deepseek_v4_flash/modal/run_dsv41_lora_2node_modal.sh"
# Run prepare_dsv41_modal.sh first. Model / data / checkpoints live on the volume (/root/data); the frozen-weight
# offload and the LoRA adapter sync dir on container-local disk ($SKYRL_LOCAL_DISK, same path on every node).
# For the 9-layer slice: MODEL_PATH=/root/data/models/DeepSeek-V4.1-Flash-bf16-chat-l0_1_2_3_20_21_22_23_24 and
# VLLM_PP_LAYER_PARTITION=4,5 (the engine's PP split must sit on the layer-20 KV-sharing group boundary).
set -x
LOCAL=${SKYRL_LOCAL_DISK:-/tmp/skyrl_local}
export MODEL_PATH=${MODEL_PATH:-/root/data/models/DeepSeek-V4.1-Flash-bf16-chat}
export DATA_DIR=${DATA_DIR:-/root/data/dapo}
export CKPT_PATH=${CKPT_PATH:-/root/data/ckpts/${RUN_NAME:-dsv41_flash_dapo_lora}}
export LORA_SYNC_PATH=${LORA_SYNC_PATH:-$LOCAL/lora_sync}
export SKYRL_FROZEN_OFFLOAD_DIR=${SKYRL_FROZEN_OFFLOAD_DIR:-$LOCAL/frozen_offload}
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-$LOCAL/vllm_autotune}
export LOGGER=${LOGGER:-wandb}
bash examples/train/deepseek_v4_flash/run_dapo_dsv41_flash_lora_sync_2node.sh "$@"
