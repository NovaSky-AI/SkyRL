#!/bin/bash
set -x

# Dummy/benchmarking full fine-tune SFT for Qwen3.8-Flash-Next (HF qwen4_exp, ~180B MoE) with
# Megatron at 131K tokens per sequence. Four nodes of 8xB200 (32 GPUs). Measures trainer throughput
# on full-context random token ids (dummy_run_full_ctx); no dataset is read.
#
#   MODEL_PATH=/path/to/Qwen3.8-Flash-Next QSA_CUDNN_WHEEL=/path/to/nvidia_cudnn_frontend-*.whl \
#     bash examples/train/sft/run_sft_dummy_qwen3_8_flash_next_megatron.sh [extra overrides...]
#
# Parallelism: TP1 x CP8 x EP8 (DP4). CP8 splits each 131K sequence over the 8 GPUs of a node, and
# EP8 keeps every expert all-to-all on NVLink inside that node; the expert gradients are reduced over
# the 4 nodes once per optimizer step. EP16 / EP32 send the all-to-all over the network instead.
# CP needs Blackwell and cuDNN frontend's QSA kernels (QSA_CUDNN_WHEEL, as in
# examples/train/qwen3_8_flash_next/run_dapo_qwen3_8_flash_next_fullft_cp8_4node.sh).
#
# Memory (peak ~152-157 GB of 178 GB per B200 at 131K):
# - MoE chunks: the MoE runs in MOE_CHUNKS sequence chunks per layer. 2 fits any routing; 1 is faster
#   but leaves no headroom on random-token batches (balanced routing gives the largest expert buffers).
# - NCCL_BUFFSIZE=524288 (half of NCCL's 4 MiB default per channel) frees several GB of
#   non-PyTorch memory; EP8 at 131K does not fit with the default.
# - The precision-aware optimizer keeps bf16 parameters plus 16-bit remainders instead of fp32 master
#   weights (same update; -7 GB).
#
# Speed:
# - overlap_param_gather + align_param_gather overlap the distributed optimizer's parameter
#   all-gather with the next step's forward.
# - gradient_accumulation_fusion accumulates weight gradients straight into main_grad (megatron's
#   apex kernel, or patches/megatron/patch_grad_accum_fusion_fallback.py without apex).
# - gdn_kernel_backend=transformer_engine and use_transformer_engine_op_fuser: TE's GDN kernels and
#   fused MoE expert MLP.
#
# One optimizer step covers BATCH_SIZE sequences (default 44 x 131,072 = 5.77M tokens, 11 micro-batches
# per DP rank). Compare settings at a realistic batch: with a handful of sequences per step the
# optimizer step dominates and hides the EP8 / overlap gains.

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/Qwen3.8-Flash-Next}"
QSA_CUDNN_WHEEL="${QSA_CUDNN_WHEEL:-}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
NUM_NODES="${NUM_NODES:-4}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NUM_STEPS="${NUM_STEPS:-4}"
BATCH_SIZE="${BATCH_SIZE:-44}"
MOE_CHUNKS="${MOE_CHUNKS:-2}"

MEGATRON_TP="${MEGATRON_TP:-1}"
MEGATRON_CP="${MEGATRON_CP:-8}"
MEGATRON_EP="${MEGATRON_EP:-8}"
if [ "$MEGATRON_CP" -gt 1 ] && [ -z "$QSA_CUDNN_WHEEL" ]; then
  echo "QSA_CUDNN_WHEEL must point at the cudnn-frontend wheel when MEGATRON_CP > 1 (split-K cuDNN QSA)." >&2
  exit 1
fi

export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-524288}"
# fla's TileLang GDN backward aborts on Blackwell.
export FLA_TILELANG="${FLA_TILELANG:-0}"

K=megatron_config.transformer_config_kwargs

uv run --isolated --extra megatron ${QSA_CUDNN_WHEEL:+--with "$QSA_CUDNN_WHEEL"} \
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
    megatron_config.context_parallel_size=$MEGATRON_CP \
    megatron_config.expert_model_parallel_size=$MEGATRON_EP \
    megatron_config.expert_tensor_parallel_size=1 \
    megatron_config.mtp_num_layers=0 \
    megatron_config.moe_token_dispatcher_type=alltoall \
    megatron_config.ddp_config.grad_reduce_in_fp32=false \
    megatron_config.ddp_config.overlap_param_gather=true \
    megatron_config.ddp_config.align_param_gather=true \
    megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true \
    $K.recompute_granularity=full \
    $K.recompute_method=uniform \
    $K.recompute_num_layers=1 \
    $K.mlp_chunks_for_training=$MOE_CHUNKS \
    $K.gradient_accumulation_fusion=true \
    $K.gdn_kernel_backend=transformer_engine \
    $K.use_transformer_engine_op_fuser=true \
    logger=console \
    project_name=skyrl_sft_benchmark \
    run_name=sft_dummy_qwen3_8_flash_next_L${MAX_LENGTH} \
    dummy_run_full_ctx=true \
    dummy_run_max_steps=$NUM_STEPS \
    "$@"
