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

uv run --isolated --env-file /home/sumanthrh/SkyRL/.env.test --extra megatron \
    python -m skyrl.train.sft_trainer \
    strategy=megatron \
    placement.num_gpus_per_node=4 \
    megatron.tensor_model_parallel_size=2 \
    megatron.pipeline_model_parallel_size=2 \
    "$@"
