#!/usr/bin/env bash
set -xeuo pipefail

export CI=true
# Prepare datasets used in tests.
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
uv run examples/train/search/searchr1_dataset.py --local_dir $HOME/data/searchR1 --split test

# Run all non-megatron tests
uv run --directory . --isolated --extra dev --extra fsdp pytest -s tests/backends/skyrl_train/gpu/gpu_ci -m "not (integrations or megatron)"

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

# Run repo-side FlashRL compatibility checks that do not require the custom wheel.
uv run --directory . --isolated --extra dev pytest -s tests/utils/test_flashrl_runtime.py

# TODO (sumanthrh): Re-enable end-to-end FlashRL GPU coverage once the
# vLLM-0.18-compatible FlashRL wheel is published.

# Run tests for new inference layer
# TODO (sumanthrh): Migrate the remaining tests: test_verifiers_generator.py , test_pause_and_continue_generation.py
_SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra dev --extra fsdp pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_policy_local_engines_e2e.py
_SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra dev --extra fsdp pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py
_SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra dev --extra fsdp pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_skyrl_gym_generator.py
_SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra dev --extra fsdp pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_lora.py
_SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra dev --extra fsdp pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_expert_parallel_inference.py
