#!/usr/bin/env bash
set -xeuo pipefail

export CI=true

# Download ShareGPT dataset
SHAREGPT_PATH="$HOME/data/ShareGPT_V3_unfiltered_cleaned_split.json"
mkdir -p "$HOME/data"
wget -q -O "$SHAREGPT_PATH" \
  "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# Run performance benchmarks
SHAREGPT_PATH="$SHAREGPT_PATH" uv run --isolated --extra dev --extra fsdp \
  pytest -s -vv tests/backends/skyrl_train/gpu/gpu_ci/benchmarks/test_benchmark_generation.py
