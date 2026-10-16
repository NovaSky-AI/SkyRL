#!/usr/bin/env bash
set -xeuo pipefail

export CI=true

# Prepare datasets used in tests (Megatron test uses gsm8k env_class).
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

# Build the 6-layer Beam slice used by the beam-6layer rows.
export SKYRL_BEAM_SLICE_PATH="${SKYRL_BEAM_SLICE_PATH:-$HOME/models/beam-6layer}"
if [ ! -f "$SKYRL_BEAM_SLICE_PATH/config.json" ]; then
  uv run --directory . --isolated --extra megatron python ci/beam_slice.py "$SKYRL_BEAM_SLICE_PATH"
fi

# Run FSDP h100 tests.
uv run --directory . --isolated --extra dev --extra fsdp pytest -s -vvv -m h100 \
    tests/backends/skyrl_train/gpu/gpu_ci/test_policy_local_engines_e2e.py

# Run Megatron h100 tests.
uv run --directory . --isolated --extra dev --extra megatron pytest -s -vvv -m h100 \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_router_replay.py \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_kimi_k25_bridge.py \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_lora_models.py \
    tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/test_sparse_mla_nope.py \
    tests/backends/skyrl_train/gpu/gpu_ci/test_trainer_full_checkpointing.py
