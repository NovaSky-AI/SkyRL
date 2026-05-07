#!/usr/bin/env bash
set -xeuo pipefail

# Run a small set of representative tests against the legacy inference path
# (`_SKYRL_USE_NEW_INFERENCE=0`). Most CI runs with the new inference layer as
# the default (since https://github.com/NovaSky-AI/SkyRL/pull/1476). This
# workflow guards against regressions in the legacy vLLM-engine-actor path
# while it still exists.
export CI=true
export _SKYRL_USE_NEW_INFERENCE=0

# Prepare datasets used in tests.
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

# FSDP-side legacy inference checks: token-based generation,
# weight-sync-then-inference round-trip, and single-turn gsm8k generation
# (both batched and async-engine paths).
uv run --directory . --isolated --extra dev --extra fsdp pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py::test_token_based_generation \
    tests/backends/skyrl_train/gpu/gpu_ci/test_save_weights_for_sampler.py::test_save_weights_for_sampler_then_inference \
    tests/backends/skyrl_train/gpu/gpu_ci/test_skyrl_gym_generator.py::test_generator_single_turn_gsm8k

# Megatron-side legacy inference check: NCCL weight sync from a Megatron
# policy into the legacy vLLM engine, then inference.
uv run --directory . --isolated --extra dev --extra megatron pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_worker.py::test_megatron_policy_weight_sync
