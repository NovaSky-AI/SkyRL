#!/bin/bash
# DeepSeek-V4.1-Flash LoRA RL shaped like a long tool-use agent workload, end to end on 16 GPUs (1 Megatron trainer
# node + 1 vLLM TP8 node), reporting trainer and sampler throughput from the same run.
#
# Per turn the policy generates exactly MAX_GENERATE_LENGTH (1,065) tokens (ignore_eos) and the environment answers
# with an OBS_TOKENS (~1,870)-token tool observation; MAX_TURNS turns per rollout. TRAIN_BATCH_SIZE x N_SAMPLES rollouts
# per step and up to MAX_STALENESS_STEPS steps in flight: 16 x 8 x (4 + 1) = 640 concurrent rollouts = 80 per vLLM GPU.
# Trainer: CP4 x DP2, EP8, LoRA r32, packed sequences, stateful full recompute, frozen base weights stored as MXFP8.
# Sampler: vLLM TP8, MXFP4 MoE + LoRA on FlashInfer's TRT-LLM kernel, 32k prefill budget, mixed-step CUDA graphs.
#
# Env (defaults in brackets): MODEL_PATH, DATA_DIR (from long_horizon_dataset.py), OUT [/root/data/skyrl-logs/<run>],
# MAX_TURNS [16], MAX_MODEL_LEN [65536], TRAIN_BATCH_SIZE [16], N_SAMPLES_PER_PROMPT [8], MAX_STALENESS_STEPS [4],
# MAX_TRAINING_STEPS [8], OFFLOAD_GB [24], RUN_NAME, CKPT_PATH. For ~200k-token rollouts: MAX_TURNS=69
# MAX_MODEL_LEN=262144 (fewer concurrent rollouts then fit the TP8 KV cache: e.g. MAX_STALENESS_STEPS=1).
set -o pipefail
cd "$(dirname "$0")/../../../.."
: "${RUN_NAME:=dsv41_depthfirst_sim}"
: "${OUT:=/root/data/skyrl-logs/$RUN_NAME}"
: "${CKPT_PATH:=/root/data/ckpts/$RUN_NAME}"
export RUN_NAME CKPT_PATH
export MAX_TURNS="${MAX_TURNS:-16}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}" N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-8}"
STALENESS="${MAX_STALENESS_STEPS:-4}"
mkdir -p "$OUT"
source examples/train_integrations/modal/bench/e2e_long_args.sh  # long-context trainer / sampler settings
export MAX_STALENESS_STEPS="$STALENESS"
RECIPE="${RECIPE:-examples/train/deepseek_v4_flash/run_async_multiturn_dsv41_flash_lora.sh}"
bash "$RECIPE" "${LONG_ARGS[@]}" \
  "generator.sampling_params.additional_kwargs={ignore_eos: true}" \
  trainer.log_path="$OUT" "$@" > "$OUT/run.log" 2>&1
rc=$?
echo "rc=$rc" >> "$OUT/run.log"
exit $rc
