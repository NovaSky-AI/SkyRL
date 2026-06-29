#!/usr/bin/env bash
set -xeuo pipefail

export CI=true

# Prepare datasets used in tests (Megatron test uses gsm8k env_class).
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

# Run FSDP h100 tests.
uv run --directory . --isolated --extra dev --extra fsdp pytest -s -vvv -m h100 \
    tests/backends/skyrl_train/gpu/gpu_ci/test_policy_local_engines_e2e.py

# Run Megatron h100 tests.
uv run --directory . --isolated --extra dev --extra megatron pytest -s -vvv -m h100 \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_router_replay.py
