#!/bin/bash
set -x

# SFT eval smoke test:
# Verifies that the new eval_dataset_name / eval_interval plumbing
# works end-to-end on the Megatron backend with the tulu3 dataset.
#
# - Short run: 10 training steps
# - Periodic eval every 5 steps -> eval fires at step 5 and step 10 (final)
# - Eval iterates in chunks of `micro_train_batch_size_per_gpu * dp_size` per dispatch
# - Small eval slice: train[:200] of allenai/tulu-3-sft-mixture (no validation split exists)
# - Console logger -- no wandb needed for a smoke test
#
# Usage:
#   NCCL_NET=Socket NCCL_NET_PLUGIN=none \
#   RAY_TMPDIR=/mnt/local_storage/ray_tmp TMPDIR=/mnt/local_storage/tmp \
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   bash examples/train/sft/run_sft_megatron_tulu3_eval_smoketest.sh

# Required: restrict to GPUs 0-3 (one NVLink island; cross-island allreduce SIGKILLs on B200)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# Required on this B200 node: NCCL must use Socket transport; gIB plugin crashes
export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
# Avoid /dev/root (96% full); use big mounted volume for Ray + general tmp
export RAY_TMPDIR=${RAY_TMPDIR:-/mnt/local_storage/ray_tmp}
export TMPDIR=${TMPDIR:-/mnt/local_storage/tmp}

mkdir -p "$RAY_TMPDIR" "$TMPDIR"

uv run --isolated --extra megatron \
    python -m skyrl.train.main_sft \
    strategy=megatron \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    dataset_name=allenai/tulu-3-sft-mixture \
    dataset_split="train[:400]" \
    messages_key=messages \
    eval_dataset_name=allenai/tulu-3-sft-mixture \
    eval_dataset_split="train[:200]" \
    eval_interval=5 \
    max_length=4096 \
    num_steps=10 \
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
    logger=console \
    project_name=skyrl_sft \
    run_name=skyrl_sft_megatron_eval_smoketest \
    ckpt_path="" \
    ckpt_interval=0 \
    resume_from="" \
    train_on_what="all_assistant_messages" \
    "$@"
