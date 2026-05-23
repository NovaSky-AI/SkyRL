#!/bin/bash
set -x

# SFT training with a `PerplexityLogger` callback.
#
# Usage:
#   bash examples/train/callbacks/run_sft_with_callbacks.sh [extra overrides...]

uv run --isolated --extra fsdp \
    -m examples.train.callbacks.main_sft_with_callbacks \
    strategy=fsdp \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    dataset_name=yahma/alpaca-cleaned \
    dataset_split="train[:200]" \
    eval_dataset_name=yahma/alpaca-cleaned \
    eval_dataset_split="train[200:240]" \
    messages_key=messages \
    max_length=512 \
    num_steps=40 \
    eval_interval=5 \
    eval_before_train=true \
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
    placement.num_gpus_per_node=1 \
    fsdp_config.cpu_offload=false \
    fsdp_config.reshard_after_forward=true \
    logger=console \
    project_name=skyrl_sft_callbacks \
    run_name=skyrl_sft_callbacks_run \
    ckpt_path="" \
    "$@"
