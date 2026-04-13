#!/bin/bash
set -x

# Dummy/benchmarking SFT training with FSDP backend for Qwen2.5-0.5B-Instruct
#
# Skips real data loading and fabricates full-context random sequences.
# Useful for profiling throughput and verifying the training pipeline.
#
# Usage:
#   bash examples/train/sft/run_sft_dummy_fsdp.sh [extra overrides...]
#
# Example:
#   bash examples/train/sft/run_sft_dummy_fsdp.sh dummy_run_max_steps=10

uv run --isolated --extra fsdp \
    python -m skyrl.train.sft_trainer \
    strategy=fsdp2 \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    max_length=2048 \
    num_steps=10 \
    batch_size=4 \
    micro_train_batch_size_per_gpu=2 \
    seed=42 \
    optimizer.lr=1e-6 \
    optimizer.weight_decay=1e-2 \
    optimizer.max_grad_norm=1.0 \
    optimizer.num_warmup_steps=0 \
    optimizer.scheduler=constant_with_warmup \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=1 \
    fsdp.cpu_offload=false \
    fsdp.reshard_after_forward=true \
    logger=console \
    project_name=skyrl_sft_benchmark \
    run_name=sft_dummy_fsdp \
    dummy_run_full_ctx=true \
    dummy_run_max_steps=5 \
    "$@"
