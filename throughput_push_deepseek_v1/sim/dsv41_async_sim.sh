#!/bin/bash
# examples/train/deepseek_v4_flash/run_async_multiturn_dsv41_flash_lora.sh on 2x8 B200: trainer on one node
# (Megatron EP8), one vLLM TP8 engine on the other, the 9-layer FP8/MXFP4 slice (KEEP_QUANT=1 conversion of release
# layers 0,1,2,3,20,21,22,23,24), gsm8k_multi_turn as the multi-turn stand-in, this branch's venv (--no-sync).
# Usage: NAME=x [STEPS=4] bash dsv41_async_sim.sh [overrides]
set -x
W=/shared/hackskyrl/tput_dsv4/SkyRL
cd $W
export PYTHONPATH=$W
NAME=${NAME:?}
LOG=/shared/hackskyrl/tput_dsv4/logs/$NAME.log
export MODEL_PATH=${MODEL_PATH:-/mnt/local_storage/hackskyrl/models/dsv41_flash_l9_q}
export DATA_DIR=/shared/hackskyrl/data/gsm8k_multi_turn
export NUM_ENGINES=${NUM_ENGINES:-1} MAX_TURNS=${MAX_TURNS:-4} MAX_GENERATE_LENGTH=${MAX_GENERATE_LENGTH:-2048}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384} TRAIN_BATCH_SIZE=${TBS:-16} N_SAMPLES_PER_PROMPT=${NSAMP:-8}
export MAX_TRAINING_STEPS=${STEPS:-4} LOGGER=console RUN_NAME=$NAME CKPT_PATH=/mnt/local_storage/hackskyrl/ckpts/$NAME
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 SKYRL_VLLM_START_PORT=${SKYRL_VLLM_START_PORT:-8600}
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/mnt/local_storage/hackskyrl/vllm_autotune
export UV_CACHE_DIR=/shared/hackskyrl/uv_cache UV_PROJECT_ENVIRONMENT=/shared/hackskyrl/uv_venvs/tput_dsv4
RECIPE=/shared/hackskyrl/tput_dsv4/recipe_async_nosync_$NAME.sh
sed 's/^uv run --isolated --extra megatron -m /uv run --no-sync --extra megatron python -m /' \
  examples/train/deepseek_v4_flash/run_async_multiturn_dsv41_flash_lora.sh > $RECIPE
bash $RECIPE "$@" > $LOG 2>&1
