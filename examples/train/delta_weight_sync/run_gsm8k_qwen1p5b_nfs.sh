#!/usr/bin/env bash
set -x

# Non-colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K
# using checkpoint-delta weight sync through a shared POSIX/NFS directory.

# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/delta_weight_sync/run_gsm8k_qwen1p5b_nfs.sh
# Refer to examples/train/delta_weight_sync/run_gsm8k_qwen1p5b_gcs.sh for the full configuration

: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${RUN_NAME:=gsm8k-qwen1p5b-delta-nfs-${RUN_ID}}"
: "${SYNC_ROOT:=/mnt/shared_storage/skyrl-delta-sync}"
: "${SYNC_DIR:=${SYNC_ROOT}/${RUN_NAME}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="$RUN_NAME" SYNC_DIR="$SYNC_DIR" \
bash "$SCRIPT_DIR/run_gsm8k_qwen1p5b_gcs.sh" "$@"
