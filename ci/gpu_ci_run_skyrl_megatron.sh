#!/usr/bin/env bash
set -xeuo pipefail

export CI=true
uv run --directory . --isolated --extra dev --extra megatron pytest -q examples/model_checks/tests
# Prepare datasets used in tests.
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# Run all megatron tests
uv run --directory . --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci -m "megatron"

