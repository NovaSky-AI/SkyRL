#!/bin/bash
# Compressed colocated RL run of the real 40-layer DeepSeek-V4.1-Flash (LoRA, R3) on 2x8 B200.
# Runs run_dapo_dsv41_flash_lora_sync_2node.sh (from NovaSky-AI/SkyRL#2573) unchanged except: 32 prompts x 12
# samples as one mini-batch (= one production mini-batch, 24 samples per DP rank), a few steps, no eval/ckpt,
# node-local paths, and this branch's shared venv. Usage: NAME=x [STEPS=3] bash dsv41_sim.sh [overrides]
set -x
W=/shared/hackskyrl/tput_dsv4/SkyRL
cd $W
export PYTHONPATH=$W
NAME=${NAME:?}
LOG=/shared/hackskyrl/tput_dsv4/logs/$NAME.log
mkdir -p /shared/hackskyrl/tput_dsv4/logs
export MODEL_PATH=${MODEL_PATH:-/mnt/local_storage/hackskyrl/models/dsv41_flash_bf16_chat}
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
bash throughput_push_deepseek_v1/sim/run_dapo_dsv41_flash_lora_sync_2node.sh \
  trainer.train_batch_size=${TBS:-32} trainer.policy_mini_batch_size=${MBS:-32} \
  generator.n_samples_per_prompt=${NSAMP:-12} \
  trainer.eval_before_train=false trainer.eval_interval=1000 trainer.ckpt_interval=1000 \
  "$@" > $LOG 2>&1
