#!/bin/bash
# DeepSeek-V4.1-Flash LoRA RL on a long multi-turn agent workload (69 turns, contexts up to ~200k tokens), fully
# async on 16 GPUs: 1 Megatron trainer node + 1 vLLM node (8 GPUs each). Wraps run_async_multiturn_dsv41_flash_lora.sh
# with the long-context trainer settings. Run from the repo root on a Ray cluster with 2 free 8-GPU nodes:
#
#   MODEL_PATH=<text-only release dir> DATA_DIR=<long_horizon_dataset.py output> \
#     bash examples/train/deepseek_v4_flash/long_horizon/run_dsv41_long_horizon.sh [overrides]
#
# (setup_dsv41_long_horizon.sh builds both.) Per turn the policy generates exactly MAX_GENERATE_LENGTH (1,065)
# tokens (ignore_eos) and the environment answers with a ~1,870-token tool observation, so 69 turns reach ~200k.
# TRAIN_BATCH_SIZE x N_SAMPLES rollouts per step and up to MAX_STALENESS_STEPS steps in flight:
# 16 x 8 x (4 + 1) = 640 concurrent rollouts = 80 per vLLM GPU.
#   trainer: TP1 PP1 CP8 EP8 (one 200k-token sequence over the node's 8 GPUs; CP4 runs out of memory on B200, fits
#            B300), packed sequences, stateful full recompute with layer inputs offloaded to pinned host memory,
#            frozen base weights stored as MXFP8, old logprobs from the training forward;
#   sampler: vLLM DP8 + EP8, MXFP4 MoE + LoRA on FlashInfer's TRT-LLM kernel, two-kernel MoE-LoRA with tuned tiles,
#            session-sticky routing to each rollout's prefix cache.
#
# Env (defaults): MODEL_PATH, DATA_DIR, OUT [$HOME/skyrl-logs/<RUN_NAME>], RUN_NAME [dsv41_long_horizon], MAX_TURNS [69],
# MAX_MODEL_LEN [262144], CP [8], VLLM_DP [8], TRAIN_BATCH_SIZE [16], N_SAMPLES_PER_PROMPT [8],
# MAX_STALENESS_STEPS [4], MAX_TRAINING_STEPS [4], OFFLOAD_GB [24: pinned host GB per trainer GPU], LOGGER [wandb].
# Shorter rollouts: MAX_TURNS=16 MAX_MODEL_LEN=65536 CP=4 (and a dataset built with --max-turns 16).
set -o pipefail
cd "$(dirname "$0")/../../../.."
: "${MODEL_PATH:?text-only DeepSeek-V4.1-Flash dir (setup_dsv41_long_horizon.sh)}"
: "${RUN_NAME:=dsv41_long_horizon}"
: "${OUT:=$HOME/skyrl-logs/$RUN_NAME}"
: "${CKPT_PATH:=$HOME/ckpts/$RUN_NAME}"
export MODEL_PATH RUN_NAME CKPT_PATH
export MAX_TURNS="${MAX_TURNS:-69}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k_long_horizon_t$MAX_TURNS}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}" N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-4}" MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-4}"
export VLLM_DP="${VLLM_DP:-8}" LOGGER="${LOGGER:-wandb}"
export ENV_CLASS=gsm8k_long_horizon MAX_GENERATE_LENGTH=1065 MAX_PROMPT_LENGTH=2048
export NUM_TRAIN_NODES=1 NUM_ENGINES=1
# Trainer memory at 200k tokens: frozen base weights kept as MXFP8, MLP inputs re-derived in the recompute, and each
# layer's input offloaded to pinned host memory (keep OFFLOAD_GB x 8 well under half the node's RAM).
export SKYRL_DSV41_COMPRESS_FROZEN_WEIGHTS=1 SKYRL_DSV41_RECOMPUTE_DERIVE_MLP_INPUT=1 SKYRL_DSV41_RECOMPUTE_OFFLOAD=1
export SKYRL_DSV41_RECOMPUTE_OFFLOAD_HOST_GB="${OFFLOAD_GB:-24}" SKYRL_DSV41_FAKE_QUANT=reference
mkdir -p "$OUT"
bash examples/train/deepseek_v4_flash/run_async_multiturn_dsv41_flash_lora.sh \
  `# trainer: context parallelism over packed (THD) sequences, contiguous layout inside the V4.1 stack` \
  "trainer.policy.megatron_config.context_parallel_size=${CP:-8}" \
  trainer.remove_microbatch_padding=true \
  trainer.policy.megatron_config.transformer_config_kwargs.attention_cp_layout=contiguous \
  trainer.policy.megatron_config.transformer_config_kwargs.linear_cp_layout=contiguous \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_kernel_backend=cudnn \
  `# trainer: stateful full recompute (needed by the frozen-weight compression and the derived MLP inputs)` \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=null \
  trainer.balance_dp_tokens=true \
  "generator.max_input_length=$MAX_MODEL_LEN" \
  "generator.sampling_params.additional_kwargs={ignore_eos: true}" \
  trainer.ckpt_interval=-1 \
  trainer.project_name=dsv41_long_horizon \
  trainer.log_path="$OUT" "$@" 2>&1 | tee "$OUT/run.log"
