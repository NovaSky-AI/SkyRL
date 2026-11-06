#!/bin/bash
# DeepSeek-V4.1-Flash LoRA RL shaped like a long tool-use agent workload, end to end on 16 GPUs (1 Megatron trainer
# node + 1 vLLM node), reporting trainer and sampler throughput from the same run.
#
# Per turn the policy generates exactly MAX_GENERATE_LENGTH (1,065) tokens (ignore_eos) and the environment answers
# with an OBS_TOKENS (~1,870)-token tool observation; MAX_TURNS (69) turns per rollout, so contexts grow to ~200k
# tokens. TRAIN_BATCH_SIZE x N_SAMPLES rollouts per step and up to MAX_STALENESS_STEPS steps in flight:
# 16 x 8 x (4 + 1) = 640 concurrent rollouts = 80 per vLLM GPU.
# Trainer: context parallel CP (8: one 200k-token sequence over the node's 8 GPUs, which fits B200), EP8, LoRA r32,
#   packed sequences, stateful full recompute, frozen base weights stored as MXFP8.
# Sampler: vLLM with data-parallel attention + expert parallelism (VLLM_DP=8: each GPU caches only its own
#   sessions' KV, ~93M tokens per B200 node), MXFP4 MoE + LoRA on FlashInfer's TRT-LLM kernel, two-kernel MoE-LoRA
#   with tuned tiles, session-sticky routing (every turn of a rollout reaches the DP rank holding its prefix cache).
#
# Env (defaults in brackets): MODEL_PATH, DATA_DIR (from long_horizon_dataset.py --max-turns MAX_TURNS), OUT
# [/root/data/skyrl-logs/<run>], MAX_TURNS [69], MAX_MODEL_LEN [262144], CP [8], VLLM_DP [8], TRAIN_BATCH_SIZE [16],
# N_SAMPLES_PER_PROMPT [8], MAX_STALENESS_STEPS [4], MAX_TRAINING_STEPS [4], OFFLOAD_GB [24], GEN_STAGGER_S [0],
# RUN_NAME, CKPT_PATH. Shorter rollouts: MAX_TURNS=16 MAX_MODEL_LEN=65536 CP=4 (and a matching dataset).
set -o pipefail
cd "$(dirname "$0")/../../../.."
: "${RUN_NAME:=dsv41_depthfirst_sim}"
: "${OUT:=/root/data/skyrl-logs/$RUN_NAME}"
: "${CKPT_PATH:=/root/data/ckpts/$RUN_NAME}"
export RUN_NAME CKPT_PATH
export MAX_TURNS="${MAX_TURNS:-69}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}" N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-4}"
export CP="${CP:-8}" VLLM_DP="${VLLM_DP:-8}"
export SKYRL_SESSION_STICKY_DIRECT="${SKYRL_SESSION_STICKY_DIRECT:-1}"
STALENESS="${MAX_STALENESS_STEPS:-4}"
mkdir -p "$OUT"
source examples/train_integrations/modal/bench/e2e_long_args.sh  # long-context trainer / sampler settings
export MAX_STALENESS_STEPS="$STALENESS"
RECIPE="${RECIPE:-examples/train/deepseek_v4_flash/run_async_multiturn_dsv41_flash_lora.sh}"
bash "$RECIPE" "${LONG_ARGS[@]}" \
  "generator.sampling_params.additional_kwargs={ignore_eos: true}" \
  trainer.fully_async.generation_worker_start_stagger_s="${GEN_STAGGER_S:-0}" \
  trainer.log_path="$OUT" "$@" > "$OUT/run.log" 2>&1
rc=$?
echo "rc=$rc" >> "$OUT/run.log"
exit $rc
