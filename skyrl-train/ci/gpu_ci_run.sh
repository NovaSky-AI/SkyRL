#!/usr/bin/env bash
set -euo pipefail

export CI=true
# Prepare datasets used in tests.
uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
uv run examples/search/searchr1_dataset.py --local_dir $HOME/data/searchR1 --split test
# Run all non-SGLang tests
uv run --directory . --isolated --extra dev --extra vllm --with deepspeed pytest -s tests/gpu/gpu_ci -m "not sglang"
# Run all SGLang tests
uv run --directory . --isolated --extra dev --extra sglang --with deepspeed pytest -s tests/gpu/gpu_ci -m "sglang"