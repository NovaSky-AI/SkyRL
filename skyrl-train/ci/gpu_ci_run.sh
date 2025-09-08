#!/usr/bin/env bash
set -euo pipefail

export CI=true
# Prepare datasets used in tests.
uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
uv run examples/search/searchr1_dataset.py --local_dir $HOME/data/searchR1 --split test
# Run all non-SGLang tests
uv run --directory . --isolated --extra dev --extra vllm --with deepspeed pytest -s tests/gpu/gpu_ci -m "not sglang" -m "not integrations"

# Run tests for "integrations" folder
uv add --active wordle --index https://hub.primeintellect.ai/will/simple/
uv run --isolated --with verifiers -- python integrations/verifiers/prepare_dataset.py --env_id will/wordle
uv run --directory . --isolated --extra dev --extra vllm --with verifiers pytest -s tests/gpu/gpu_ci/ -m "integrations"

# Run all SGLang tests
uv run --directory . --isolated --extra dev --extra sglang --with deepspeed pytest -s tests/gpu/gpu_ci -m "sglang"