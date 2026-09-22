#!/usr/bin/env bash
set -x

# Non-colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K
# using checkpoint-delta weight sync through a shared POSIX/NFS directory.

# Run:
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/delta_weight_sync/run_gsm8k_qwen1p5b_s3.sh
# 
# Refer to examples/train/delta_weight_sync/run_gsm8k_qwen1p5b_gcs.sh for the full configuration

: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${RUN_NAME:=gsm8k-qwen1p5b-delta-s3-${RUN_ID}}"
: "${SYNC_DIR:?Set SYNC_DIR to a unique s3:// path for this run}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The `aws` extra provides the s5cmd CLI that the delta backend shells out to for s3://.
RUN_NAME="$RUN_NAME" SYNC_DIR="$SYNC_DIR" CLOUD_EXTRA=aws \
bash "$SCRIPT_DIR/run_gsm8k_qwen1p5b_gcs.sh" "$@"
