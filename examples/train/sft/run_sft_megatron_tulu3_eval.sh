#!/bin/bash
set -x

# SFT training with Megatron backend for Qwen2.5-0.5B-Instruct on Tulu3 with
# periodic eval-loss reporting.
#
# Based on run_sft_megatron_tulu3_30k.sh, with the new eval_dataset_* plumbing:
# - Train slice: train[:30000] of allenai/tulu-3-sft-mixture
# - Eval slice:  train[-500:] (last 500 examples; held-out from training)
# - eval_interval=50  -> eval fires at steps 50, 100, 150, 200 (+final)
# - eval iterates in chunks of `micro_train_batch_size_per_gpu * dp_size` per dispatch
# - logger=wandb so eval_loss is plotted alongside train loss
#
# num_steps=200 keeps the run bounded (~few minutes per eval cadence) while
# still letting us watch eval_loss decrease over multiple eval checkpoints.
#
# B200-specific knobs (inherited from the 30k script):
# - CUDA_VISIBLE_DEVICES=0,1,2,3 (one NVLink island; cross-island allreduce SIGKILLs)
# - NCCL_NET=Socket / NCCL_NET_PLUGIN=none (gIB plugin crashes)
# - micro_train_batch_size_per_gpu=6 -> num_microbatches=24/(4*6)=1 (>1 hangs Megatron)
# - use_sample_packing=true (REQUIRED -- packed flash-attn path is the stable one on B200)
# - max_length=4096 (no truncation; dataset max=16236 tokens, mean=360 tokens)
# - train_on_what=all_assistant_messages (LLaMA-Factory default)
#
# Usage:
#   NCCL_NET=Socket NCCL_NET_PLUGIN=none \
#   RAY_TMPDIR=/mnt/local_storage/ray_tmp TMPDIR=/mnt/local_storage/tmp \
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   bash examples/train/sft/run_sft_megatron_tulu3_eval.sh [extra overrides...]

# Required: restrict to GPUs 0-3 (one NVLink island; cross-island allreduce SIGKILLs on B200)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# Required on this B200 node: NCCL must use Socket transport; gIB plugin crashes
export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
# Avoid /dev/root (near full); use big mounted volume for Ray + general tmp
export RAY_TMPDIR=${RAY_TMPDIR:-/mnt/local_storage/ray_tmp}
export TMPDIR=${TMPDIR:-/mnt/local_storage/tmp}

mkdir -p "$RAY_TMPDIR" "$TMPDIR"

uv run --isolated --extra megatron \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    dataset_name=allenai/tulu-3-sft-mixture \
    dataset_split="train[:30000]" \
    messages_key=messages \
    eval_dataset_name=allenai/tulu-3-sft-mixture \
    eval_dataset_split="train[-500:]" \
    eval_interval=50 \
    max_length=4096 \
    num_steps=200 \
    batch_size=24 \
    micro_train_batch_size_per_gpu=6 \
    use_sample_packing=true \
    seed=42 \
    optimizer_config.lr=1e-6 \
    optimizer_config.weight_decay=1e-2 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=0 \
    optimizer_config.scheduler=constant_with_warmup \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=4 \
    megatron_config.tensor_model_parallel_size=1 \
    megatron_config.pipeline_model_parallel_size=1 \
    megatron_config.context_parallel_size=1 \
    logger=wandb \
    project_name=skyrl-sft-eval \
    run_name=skyrl_sft_megatron_tulu3_eval \
    ckpt_path="" \
    ckpt_interval=0 \
    resume_from="" \
    train_on_what="all_assistant_messages" \
    "$@"
