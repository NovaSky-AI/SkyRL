#!/usr/bin/env bash
set -x

# Compatibility wrapper. New delta weight sync examples live under
# examples/train/delta_weight_sync.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/../delta_weight_sync/run_gsm8k_qwen1p5b_gcs.sh" "$@"
