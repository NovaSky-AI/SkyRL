#!/usr/bin/env bash
set -x

# Non-colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K
# using checkpoint-delta weight sync through a shared POSIX/NFS directory.

: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${RUN_NAME:=gsm8k-qwen1p5b-delta-nfs-${RUN_ID}}"
: "${SYNC_ROOT:=/mnt/shared_storage/skyrl-delta-sync}"
: "${SYNC_DIR:=${SYNC_ROOT}/${RUN_NAME}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="$RUN_NAME" SYNC_DIR="$SYNC_DIR" \
bash "$SCRIPT_DIR/run_gsm8k_qwen1p5b_gcs.sh" "$@"
