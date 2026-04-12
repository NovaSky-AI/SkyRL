#!/bin/bash
set -x

# SFT training with FSDP backend for Qwen2.5-0.5B-Instruct
#
# This script runs supervised fine-tuning using FSDP on 1 GPU with the
# Alpaca dataset.
#
# Usage:
#   bash examples/train/sft/run_sft_fsdp.sh [extra overrides...]
#
# Example:
#   bash examples/train/sft/run_sft_fsdp.sh num_steps=20 batch_size=8

uv run --isolated --env-file /home/sumanthrh/SkyRL/.env.test --extra fsdp \
    python -m skyrl.train.sft_trainer \
    strategy=fsdp2 \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    placement.num_gpus_per_node=1 \
    "$@"
