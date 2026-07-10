#!/bin/bash
set -x

# VLM SFT training with the Megatron backend for Qwen3-VL-2B-Instruct.
#
# Runs supervised fine-tuning of a vision-language model on a small subset of
# HuggingFaceM4/the_cauldron with pure data parallelism (DP=4) on 4 GPUs.
# For larger models, increase TP/PP below (e.g. tensor_model_parallel_size=2,
# pipeline_model_parallel_size=2).
#
# VLM constraints (mirrored from the FSDP VLM path; 3D RoPE + image token
# positions, see skyrl/backends/skyrl_train/workers/model_wrapper.py):
#   - remove_microbatch_padding must be false (no microbatch padding / packing)
#   - sequence_parallel_size and context_parallel_size must be 1
#   - train_on_what must be last_assistant_message
#
# Usage:
#   uv run examples/train/sft/prepare_cauldron_vlm.py --output-dir $HOME/data/cauldron-vlm
#   bash examples/train/sft/run_sft_megatron_vlm.sh
#
# Example:
#   bash examples/train/sft/run_sft_megatron_vlm.sh num_steps=20 batch_size=8

: "${DATA_DIR:="$HOME/data/cauldron-vlm"}"
: "${CKPT_DIR:="$HOME/ckpts/skyrl_sft_megatron_vlm"}"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "=== Generating the_cauldron VLM SFT dataset ==="
  uv run examples/train/sft/prepare_cauldron_vlm.py --output-dir "$DATA_DIR"
fi

uv run --isolated --extra megatron \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path=Qwen/Qwen3-VL-2B-Instruct \
    "train_datasets=[$DATA_DIR]" \
    "train_dataset_splits=[train]" \
    messages_key=messages \
    max_length=4096 \
    num_steps=30 \
    batch_size=4 \
    micro_train_batch_size_per_gpu=1 \
    remove_microbatch_padding=false \
    train_on_what=last_assistant_message \
    seed=42 \
    optimizer_config.lr=2e-6 \
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
    run_name=skyrl_sft_megatron_vlm \
    ckpt_path="$CKPT_DIR" \
    ckpt_interval=10 \
    hf_save_interval=20 \
    resume_from="" \
    "$@"
