#!/usr/bin/env bash
set -xeuo pipefail

export CI=true
# Prepare datasets used in tests.
uv run --isolated --extra megatron examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"

test_root=tests/backends/skyrl_train/gpu/gpu_ci
case "${1:-all}" in
    megatron_dir)
        uv run --isolated --extra dev --extra megatron pytest -s "$test_root/megatron" -m megatron
        ;;
    other)
        uv run --isolated --extra dev --extra megatron pytest -s "$test_root" -m megatron --ignore="$test_root/megatron"
        ;;
    all)
        uv run --isolated --extra dev --extra megatron pytest -s "$test_root" -m megatron
        ;;
    *)
        echo "Unknown Megatron CI shard: $1" >&2
        exit 2
        ;;
esac
