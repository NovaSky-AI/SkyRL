#!/bin/bash
set -xeou pipefail

# SFT training with Megatron backend for Qwen2.5-1.5B-Instruct on a
# tool-calling dataset (Salesforce/APIGen-MT-5k).
#
# Usage:
# 
# export DATA_DIR=$HOME/data/apigen-mt-5k-openai 
# uv run examples/train/sft/prepare_apigen_mt.py --output_dir $DATA_DIR
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/sft/run_sft_megatron_apigen_mt.sh num_epochs=1 num_steps=<num_steps>

: "${DATA_DIR:="$HOME/data/apigen-mt-5k-openai"}"
: "${CKPT_PATH:="$HOME/ckpts/skyrl_apigen_mt"}"

uv run --isolated --extra megatron --python 3.12 \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path=Qwen/Qwen2.5-1.5B-Instruct \
    dataset_name="$DATA_DIR" \
    dataset_split="train[:4000]" \
    eval_dataset_name="$DATA_DIR" \
    eval_dataset_split="train[4000:]" \
    messages_key=messages \
    tools_key=tools \
    system_key=system \
    max_length=8192 \
    num_steps=50 \
    batch_size=8 \
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
    megatron_config.tensor_model_parallel_size=1 \
    megatron_config.pipeline_model_parallel_size=1 \
    megatron_config.context_parallel_size=1 \
    logger=console \
    project_name=skyrl_sft \
    run_name=skyrl_sft_megatron_apigen_mt \
    ckpt_path="$CKPT_PATH" \
    ckpt_interval=0 \
    resume_from="" \
    train_on_what="all_assistant_messages" \
    "$@"
