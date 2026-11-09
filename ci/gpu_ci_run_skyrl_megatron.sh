#!/usr/bin/env bash
set -xeuo pipefail

export CI=true
# Prepare datasets used in tests.
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# Split the suite so each Anyscale job fits within GitHub's six-hour limit.
TEST_ROOT=tests/backends/skyrl_train/gpu/gpu_ci
case "${1:-all}" in
    megatron_dir)
        uv run --directory . --isolated --extra dev --extra megatron pytest -s "$TEST_ROOT/megatron" -m "megatron"
        ;;
    other)
        uv run --directory . --isolated --extra dev --extra megatron pytest -s "$TEST_ROOT" -m "megatron" --ignore="$TEST_ROOT/megatron"
        ;;
    all)
        uv run --directory . --isolated --extra dev --extra megatron pytest -s "$TEST_ROOT" -m "megatron"
        ;;
    *)
        echo "Unknown Megatron CI shard: $1" >&2
        exit 2
        ;;
esac
