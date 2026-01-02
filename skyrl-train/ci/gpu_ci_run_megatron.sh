#!/usr/bin/env bash
set -xeuo pipefail

export CI=true
# Prepare datasets used in tests.
uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# run all megatron tests
uv run --directory . --isolated --extra dev --extra mcore pytest -s tests/gpu/gpu_ci -m "megatron"
