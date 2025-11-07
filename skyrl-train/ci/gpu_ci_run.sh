#!/usr/bin/env bash
set -xeuo pipefail

export CI=true
# Prepare datasets used in tests.
uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
uv run examples/search/searchr1_dataset.py --local_dir $HOME/data/searchR1 --split test
# Run all non-SGLang tests
# TODO: enable megatron when tests and dependencies are fixed
uv run --directory . --isolated --extra dev --extra vllm --extra deepspeed pytest -s tests/gpu/gpu_ci -m "not (sglang or integrations or megatron)"

# Run tests for "integrations" folder
if add_integrations=$(uv add --active wordle --index https://hub.primeintellect.ai/will/simple/ 2>&1); then
    echo "Running integration tests"
    uv run --isolated --with verifiers -- python integrations/verifiers/prepare_dataset.py --env_id will/wordle
    uv run --directory . --isolated --extra dev --extra vllm --with verifiers pytest -s tests/gpu/gpu_ci/ -m "integrations"
else 
    echo "Skipping integrations tests. Failed to execute uv add command"
    echo "$add_integrations"
fi

# Run all SGLang tests
uv run --directory . --isolated --extra dev --extra sglang --extra deepspeed pytest -s tests/gpu/gpu_ci -m "sglang"