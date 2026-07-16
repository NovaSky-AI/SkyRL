#!/usr/bin/env bash
set -xeuo pipefail

export CI=true
# Prepare datasets used in tests.
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
uv run examples/train/search/searchr1_dataset.py --local_dir $HOME/data/searchR1 --split test

# Run all non-megatron tests
uv run --directory . --isolated --extra dev --extra fsdp pytest -s tests/backends/skyrl_train/gpu/gpu_ci -m "not (integrations or megatron)" --ignore=tests/backends/skyrl_train/gpu/gpu_ci/megatron

## TODO: enable integrations
# # Run tests for "integrations" folder
# if add_integrations=$(uv add --active wordle --index https://hub.primeintellect.ai/will/simple/ 2>&1); then
#     echo "Running integration tests"
#     uv run --isolated --with verifiers@git+https://github.com/PrimeIntellect-ai/verifiers.git@15f68 -- python integrations/verifiers/prepare_dataset.py --env_id will/wordle
#     uv run --directory . --isolated --extra dev --extra vllm --with verifiers@git+https://github.com/PrimeIntellect-ai/verifiers.git@15f68 pytest -s tests/gpu/gpu_ci/ -m "integrations"
# else 
#     echo "Skipping integrations tests. Failed to execute uv add command"
#     echo "$add_integrations"
# fi
