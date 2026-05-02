#!/bin/bash
set -x


# SFT training with Megatron backend for Qwen2.5-0.5B-Instruct on Tulu3
#
# This script runs supervised fine-tuning using the Megatron backend with
# pure data parallelism (DP=4) on 4 GPUs with the Tulu3 dataset.
# For larger models that exceed single-GPU memory, increase TP/PP below
# (e.g. tensor_model_parallel_size=2, pipeline_model_parallel_size=2).

# Usage:
#   bash examples/train/sft/run_sft_megatron_tulu3_50k.sh [extra overrides...]

: "${CKPT_PATH:="$HOME/ckpts/skyrl_tulu3_50k"}"
: "${export_path:="/tmp/skyrl_tulu3_50k_hf_ckpts"}"

uv run --isolated --extra megatron \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    dataset_name=allenai/tulu-3-sft-mixture \
    dataset_split="train[:50000]" \
    messages_key=messages \
    max_length=4096 \
    num_steps=4166 \
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
    project_name=skyrl_sft \
    run_name=skyrl_sft_megatron_tulu3_50k \
    ckpt_path="$CKPT_PATH" \
    ckpt_interval=0 \
    resume_from="" \
    train_on_what="all_assistant_messages" \
    hf_save_interval=2083 \
    export_path="$export_path" \
    "$@"
