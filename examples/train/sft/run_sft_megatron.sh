#!/bin/bash
set -x

# SFT training with Megatron backend for Qwen3-0.6B
#
# This script runs supervised fine-tuning using Megatron's tensor and pipeline
# parallelism (TP=2, PP=2) on 4 GPUs with the Alpaca dataset.
#
# Usage:
#   bash examples/train/sft/run_sft_megatron.sh [extra overrides...]
#
# Example:
#   bash examples/train/sft/run_sft_megatron.sh num_steps=20 batch_size=8

uv run --isolated --extra megatron \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path=Qwen/Qwen3-0.6B \
    dataset_name=yahma/alpaca-cleaned \
    dataset_split="train[:100]" \
    messages_key=messages \
    max_length=512 \
    num_steps=10 \
    batch_size=4 \
    micro_train_batch_size_per_gpu=2 \
    seed=42 \
    optimizer_config.lr=1e-6 \
    optimizer_config.weight_decay=1e-2 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=0 \
    optimizer_config.scheduler=constant_with_warmup \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=4 \
    megatron_config.tensor_model_parallel_size=2 \
    megatron_config.pipeline_model_parallel_size=2 \
    megatron_config.context_parallel_size=1 \
    logger=console \
    project_name=skyrl_sft \
    run_name=skyrl_sft_megatron_run \
    ckpt_path="" \
    ckpt_interval=0 \
    resume_from="" \
    "$@"
