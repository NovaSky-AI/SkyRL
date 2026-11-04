#!/bin/bash
# Compressed colocated RL run of DeepSeek-V4.1-Flash (LoRA, R3) on 2x8 B200 with the layout of
# examples/train/deepseek_v4_flash/run_dapo_dsv41_flash_lora_sync_2node.sh (Megatron EP16 TP1, one vLLM engine TP8 x PP2).
# Runs that recipe unchanged except: 32 prompts x 12 samples as one mini-batch (= one production mini-batch, 24
# samples per DP rank), a few steps, no eval/ckpt, node-local paths, and this branch's shared venv (--no-sync).
#
# Default model: 9-layer slice with all 384 experts and every V4.1 layer role (layers 0,1,2,3,20,21,22,23,24 of the
# release), built on each node by examples/train/deepseek_v4_flash/dev/convert_dsv41_bf16_parallel.py. Its PP split
# must land on the layer-20 KV-sharing group boundary: VLLM_PP_LAYER_PARTITION=4,5. The full 40-layer model does not
# fit these hosts' RAM (vLLM's sleep backup + Megatron's load exceed 2 TB on a node; see TRACKING.md):
#   MODEL_PATH=/mnt/local_storage/hackskyrl/models/dsv41_flash_bf16_chat VLLM_PP_LAYER_PARTITION= NAME=x bash dsv41_sim.sh
#
# Usage: NAME=x [STEPS=3] [TBS=32 MBS=32 NSAMP=12] bash dsv41_sim.sh [overrides]
set -x
W=/shared/hackskyrl/tput_dsv4/SkyRL
cd $W
export PYTHONPATH=$W
NAME=${NAME:?}
LOG=/shared/hackskyrl/tput_dsv4/logs/$NAME.log
mkdir -p /shared/hackskyrl/tput_dsv4/logs
export MODEL_PATH=${MODEL_PATH:-/mnt/local_storage/hackskyrl/models/dsv41_flash_l9_bf16_chat}
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION-4,5}
[ -z "$VLLM_PP_LAYER_PARTITION" ] && unset VLLM_PP_LAYER_PARTITION
export DATA_DIR=/shared/hackskyrl/data/dapo
export LOGGER=console RUN_NAME=$NAME CKPT_PATH=/mnt/local_storage/hackskyrl/ckpts/$NAME
export LORA_SYNC_PATH=/mnt/local_storage/hackskyrl/lora_sync
export SKYRL_FROZEN_OFFLOAD_DIR=${SKYRL_FROZEN_OFFLOAD_DIR:-/mnt/local_storage/hackskyrl/frozen_offload}
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/mnt/local_storage/hackskyrl/vllm_autotune
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
export SKYRL_VLLM_START_PORT=${SKYRL_VLLM_START_PORT:-8500}
export UV_CACHE_DIR=/shared/hackskyrl/uv_cache
export UV_PROJECT_ENVIRONMENT=/shared/hackskyrl/uv_venvs/tput_dsv4
export MAX_TRAINING_STEPS=${STEPS:-3}
RECIPE=/shared/hackskyrl/tput_dsv4/recipe_nosync_$NAME.sh
sed 's/^uv run --isolated --extra megatron -m /uv run --no-sync --extra megatron python -m /' \
  examples/train/deepseek_v4_flash/run_dapo_dsv41_flash_lora_sync_2node.sh > $RECIPE
bash $RECIPE \
  trainer.train_batch_size=${TBS:-32} trainer.policy_mini_batch_size=${MBS:-32} \
  generator.n_samples_per_prompt=${NSAMP:-12} \
  trainer.eval_before_train=false trainer.eval_interval=1000 trainer.ckpt_interval=1000 \
  "$@" > $LOG 2>&1
