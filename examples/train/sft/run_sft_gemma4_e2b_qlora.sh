#!/bin/bash
set -x

# Text-only SFT of Gemma 4 E2B with a 4-bit base model and LoRA adapters.
# Accept the Gemma license on Hugging Face before running the Google checkpoint.

uv run --isolated --extra fsdp --extra qlora \
    python -m skyrl.train.main_sft \
    strategy=fsdp \
    model.path=google/gemma-4-E2B-it \
    model.bitsandbytes_4bit.enabled=true \
    model.lora.rank=8 \
    model.lora.alpha=16 \
    model.lora.target_modules=all-linear \
    train_datasets="['yahma/alpaca-cleaned']" \
    train_dataset_splits="['train[:100]']" \
    messages_key=messages \
    max_length=128 \
    num_steps=10 \
    batch_size=1 \
    micro_train_batch_size_per_gpu=1 \
    remove_microbatch_padding=false \
    flash_attn=false \
    seed=42 \
    optimizer_config.lr=2e-5 \
    optimizer_config.weight_decay=0.0 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=0 \
    optimizer_config.scheduler=constant_with_warmup \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=1 \
    fsdp_config.cpu_offload=false \
    fsdp_config.reshard_after_forward=true \
    fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Gemma4Model']" \
    logger=console \
    project_name=skyrl_sft_qlora \
    run_name=gemma4_e2b_qlora \
    ckpt_path="" \
    ckpt_interval=0 \
    resume_from="" \
    "$@"
