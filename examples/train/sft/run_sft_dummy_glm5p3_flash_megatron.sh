#!/bin/bash
set -x

# Dummy/benchmarking full fine-tune SFT for GLM-5.3-Flash with Megatron (random token ids).
# Default: 8 nodes x 8xB200. Parallelism and memory knobs follow
# examples/train/glm5_3_flash/run_dapo_glm5p3_flash_fullft_sync_8node.sh.
#
# Weights are randomly initialized (SKYRL_MEGATRON_RANDOM_INIT=1), so MODEL_PATH only needs the
# HF config.json + tokenizer files, present at the same path on every node. Weight values don't
# affect memory or throughput on random tokens.
#
# MODEL_PATH: a directory with zai-org/GLM-5.3-Flash's config.json (drop its "quantization_config")
# and tokenizer files (tokenizer.json, tokenizer_config.json, chat_template.jinja).
#
# Usage:
#   MODEL_PATH=/path/to/glm5p3_flash_cfg MAX_LENGTH=8192 \
#       bash examples/train/sft/run_sft_dummy_glm5p3_flash_megatron.sh [extra overrides...]
#
# Defaults fit up to 16k tokens per sequence. 32k needs TP8 and more selective recompute
# (measured: ~127 GB allocated per GPU, ~2.9k tokens/s on 64 B200s):
#   MEGATRON_TP=8 RECOMPUTE_MODULES='[core_attn,moe,moe_act,mla_up_proj,shared_experts]' MAX_LENGTH=32768 ...

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a GLM-5.3-Flash config + tokenizer directory}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
NUM_NODES="${NUM_NODES:-8}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NUM_STEPS="${NUM_STEPS:-4}"

# KDA has no context-parallel path and megatron-core rejects mHC with PP>1, so TP is the only
# dense-side divisor. dp = 64 / TP4 = 16; one full-length sequence per DP rank per step.
MEGATRON_TP="${MEGATRON_TP:-4}"
MEGATRON_EP="${MEGATRON_EP:-32}"
MEGATRON_ETP="${MEGATRON_ETP:-2}"
DP=$(( NUM_NODES * NUM_GPUS_PER_NODE / MEGATRON_TP ))
BATCH_SIZE="${BATCH_SIZE:-$DP}"

RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-selective}"
RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-[core_attn,moe]}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-null}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-null}"
# Fused TileLang SparseMLA for the DSA layers. glm5_next/dsa.py pads GLM-5.3's NoPE-MLA layout
# into the kernel's; without a backend megatron-core falls back to dense [heads, sq, sq] FP32
# scores (128 GiB/GPU at 32k with 32 local heads).
DSA_KERNEL_BACKEND="${DSA_KERNEL_BACKEND:-tilelang}"

export SKYRL_MEGATRON_RANDOM_INIT=1
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export FLA_TILELANG=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

uv run --isolated --extra megatron \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path="$MODEL_PATH" \
    language_model_only=true \
    max_length=$MAX_LENGTH \
    batch_size=$BATCH_SIZE \
    micro_train_batch_size_per_gpu=1 \
    remove_microbatch_padding=true \
    fused_lm_head_logprob=true \
    logprobs_chunk_size=1024 \
    seed=42 \
    optimizer_config.lr=1e-6 \
    optimizer_config.weight_decay=0.1 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=0 \
    optimizer_config.scheduler=constant_with_warmup \
    placement.num_nodes=$NUM_NODES \
    placement.num_gpus_per_node=$NUM_GPUS_PER_NODE \
    megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
    megatron_config.pipeline_model_parallel_size=1 \
    megatron_config.context_parallel_size=1 \
    megatron_config.expert_model_parallel_size=$MEGATRON_EP \
    megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
    megatron_config.mtp_num_layers=0 \
    megatron_config.moe_grouped_gemm=true \
    megatron_config.moe_token_dispatcher_type=alltoall \
    megatron_config.moe_router_score_function=sigmoid \
    megatron_config.moe_router_load_balancing_type=none \
    megatron_config.transformer_config_kwargs.sequence_parallel=true \
    megatron_config.transformer_config_kwargs.recompute_granularity=$RECOMPUTE_GRANULARITY \
    megatron_config.transformer_config_kwargs.recompute_modules=$RECOMPUTE_MODULES \
    megatron_config.transformer_config_kwargs.recompute_method=$RECOMPUTE_METHOD \
    megatron_config.transformer_config_kwargs.recompute_num_layers=$RECOMPUTE_NUM_LAYERS \
    megatron_config.transformer_config_kwargs.mlp_chunks_for_training=64 \
    megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
    megatron_config.transformer_config_kwargs.disable_parameter_transpose_cache=true \
    megatron_config.transformer_config_kwargs.dsa_kernel_backend=$DSA_KERNEL_BACKEND \
    megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
    megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
    megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
    megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
    logger=console \
    project_name=skyrl_sft_benchmark \
    run_name=sft_dummy_glm5p3_flash_L${MAX_LENGTH} \
    dummy_run_full_ctx=true \
    dummy_run_max_steps=$NUM_STEPS \
    "$@"
