#!/bin/bash
# Trainer-only RL simulation on one 8-GPU node: the real Megatron RL training step (policy logprob forward,
# GRPO advantages, policy forward/backward + optimizer) on dummy rollouts, with no vLLM and no weight sync
# (`examples/train_scripts/full_context`, `trainer.skip_inference_engines=true`).
#
# Rollouts: per step PROMPTS x SAMPLES trajectories; prompts 128-512 tokens; responses uniform in
# [512, RESP] with 25% truncated at RESP; token ids are windows of tokenized dataset prompts (real-text
# routing). Multi-turn: each response is TURNS assistant spans with OBS-token loss-masked observations
# between them. R3: synthetic uniform routes (`trainer.dummy_routes`) replayed by the router.
# Weights: random init (SKYRL_MEGATRON_RANDOM_INIT=1), so only the model's config is read.
#
# Usage: NAME=x [MODEL_PATH= TP= EP= RESP= TURNS= OBS= PROMPTS= SAMPLES= STEPS= R3= MTPM= PACK=] \
#        bash trainer_sim.sh [extra overrides...]
set -x
SKYRL_DIR="${SKYRL_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$SKYRL_DIR"
export PYTHONPATH="$SKYRL_DIR"
NAME="${NAME:?}"
MODEL_PATH="${MODEL_PATH:-/shared/hackskyrl/models/dsv41_flash_l8_e64}"
DATA_DIR="${DATA_DIR:-/shared/hackskyrl/data/gsm8k_multi_turn}"
TP="${TP:-1}"; EP="${EP:-8}"
RESP="${RESP:-16384}"; TURNS="${TURNS:-4}"; OBS="${OBS:-256}"
PROMPTS="${PROMPTS:-16}"; SAMPLES="${SAMPLES:-8}"; STEPS="${STEPS:-3}"
R3="${R3:-true}"
MTPM="${MTPM:--1}"  # trainer.max_tokens_per_microbatch (-1: off; token batching needs packing)
PACK="${PACK:-false}"  # trainer.remove_microbatch_padding (THD packing; DSv4 attention rejects it)
MAX_LEN=$(( 512 + RESP ))
export SKYRL_MEGATRON_RANDOM_INIT="${SKYRL_MEGATRON_RANDOM_INIT:-1}" SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export UV_CACHE_DIR=/shared/hackskyrl/uv_cache UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/shared/hackskyrl/uv_venvs/tput_dsv4}"
uv run --no-sync --extra megatron python -m examples.train_scripts.full_context.main_full_ctx \
  data.train_data="['$DATA_DIR/train.parquet']" data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.num_dummy_steps=$STEPS trainer.skip_inference_engines=true \
  trainer.dummy_length_distribution=mixture trainer.dummy_min_prompt_length=128 \
  trainer.dummy_max_prompt_length=512 trainer.dummy_min_response_length=512 \
  trainer.dummy_truncated_fraction=0.25 trainer.dummy_token_source=dataset \
  trainer.dummy_num_turns=$TURNS trainer.dummy_observation_length=$OBS trainer.dummy_routes=$R3 \
  trainer.algorithm.advantage_estimator=grpo trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.policy.model.path="$MODEL_PATH" trainer.strategy=megatron \
  trainer.placement.colocate_all=false trainer.placement.policy_num_gpus_per_node=8 \
  trainer.policy.megatron_config.tensor_model_parallel_size=$TP \
  trainer.policy.megatron_config.expert_model_parallel_size=$EP \
  trainer.policy.megatron_config.moe_enable_routing_replay=$R3 \
  generator.inference_engine.enable_return_routed_experts=$R3 \
  generator.max_input_length=512 generator.sampling_params.max_generate_length=$RESP \
  generator.n_samples_per_prompt=$SAMPLES \
  trainer.max_prompt_length=512 trainer.algorithm.max_seq_len=$MAX_LEN \
  trainer.train_batch_size=$PROMPTS trainer.policy_mini_batch_size=$PROMPTS \
  trainer.micro_forward_batch_size_per_gpu=1 trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_tokens_per_microbatch=$MTPM trainer.remove_microbatch_padding=$PACK \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=selective \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[moe,mhc,layernorm] \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.logger=console trainer.project_name=throughput_push_v1_sim trainer.run_name=$NAME "$@"
