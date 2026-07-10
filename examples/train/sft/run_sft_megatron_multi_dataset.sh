#!/bin/bash
set -x

# SFT training on a weighted mixture of multiple datasets (Megatron backend).
#
# Trains on Tulu3 + Alpaca-Cleaned mixed 80/20 per batch (independent of the
# dataset sizes) via the core DataMixingSampler, and evaluates on held-out
# slices of both datasets separately. Eval metrics are logged per dataset under
# eval/{name}/loss (names from eval_dataset_names).
#
# Note the quoting: OmegaConf list overrides must be a single shell argument.
#
# Usage:
#   bash examples/train/sft/run_sft_megatron_multi_dataset.sh [extra overrides...]
#
# Example:
#   bash examples/train/sft/run_sft_megatron_multi_dataset.sh num_steps=20 'train_dataset_weights=[0.5,0.5]'

uv run --isolated --extra megatron \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    "train_datasets=[allenai/tulu-3-sft-mixture,yahma/alpaca-cleaned]" \
    "train_dataset_splits=[train[:800],train[:200]]" \
    "train_dataset_weights=[0.8,0.2]" \
    "eval_datasets=[allenai/tulu-3-sft-mixture,yahma/alpaca-cleaned]" \
    "eval_dataset_splits=[train[-100:],train[200:300]]" \
    "eval_dataset_names=[tulu3,alpaca]" \
    eval_interval=5 \
    messages_key=messages \
    max_length=512 \
    num_steps=10 \
    batch_size=4 \
    micro_train_batch_size_per_gpu=1 \
    remove_microbatch_padding=true \
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
    logger=console \
    project_name=skyrl_sft \
    run_name=skyrl_sft_megatron_multi_dataset_run \
    ckpt_path="" \
    ckpt_interval=0 \
    resume_from="" \
    "$@"
