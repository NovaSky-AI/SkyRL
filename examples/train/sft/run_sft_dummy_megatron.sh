#!/bin/bash
set -x

# Dummy/benchmarking SFT training with Megatron backend for Qwen3-0.6B
#
# Skips real data loading and fabricates full-context random sequences.
# Useful for profiling throughput and verifying the training pipeline.
#
# Usage:
#   bash examples/train/sft/run_sft_dummy_megatron.sh [extra overrides...]
#
# Example:
#   bash examples/train/sft/run_sft_dummy_megatron.sh dummy_run_max_steps=10

uv run --isolated --extra megatron \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path=Qwen/Qwen3-0.6B \
    max_length=2048 \
    num_steps=10 \
    batch_size=4 \
    micro_train_batch_size_per_gpu=2 \
    use_sample_packing=true \
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
    project_name=skyrl_sft_benchmark \
    run_name=sft_dummy_megatron \
    dummy_run_full_ctx=true \
    dummy_run_max_steps=5 \
    "$@"
