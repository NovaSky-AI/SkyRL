#!/usr/bin/env bash
set -xeuo pipefail

export CI=true

# Explicitly set CUDA_VISIBLE_DEVICES to make all GPUs available to the tests.
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')

# Prepare datasets used in tests.
uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
uv run examples/search/searchr1_dataset.py --local_dir $HOME/data/searchR1 --split test
# Run all non-SGLang tests sequentially using `-n 0` to prevent resource contention.
uv run --directory . --isolated --extra dev --extra vllm --extra deepspeed pytest -s tests/gpu/gpu_ci -m "not (sglang or integrations)" -n 0

# Run tests for "integrations" folder
if add_integrations=$(uv add --active wordle --index https://hub.primeintellect.ai/will/simple/ 2>&1); then
    echo "Running integration tests"
    uv run --isolated --with verifiers -- python integrations/verifiers/prepare_dataset.py --env_id will/wordle
    uv run --directory . --isolated --extra dev --extra vllm --with verifiers pytest -s tests/gpu/gpu_ci/ -m "integrations" -n 0
else 
    echo "Skipping integrations tests. Failed to execute uv add command"
    echo "$add_integrations"
fi

# Run all SGLang tests
uv run --directory . --isolated --extra dev --extra sglang --with deepspeed pytest -s tests/gpu/gpu_ci -m "sglang" -n 0