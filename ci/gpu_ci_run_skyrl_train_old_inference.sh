#!/usr/bin/env bash
set -xeuo pipefail

# Guard against regressions in the legacy vLLM-engine-actor path; most CI
# runs with the new inference layer (_SKYRL_USE_NEW_INFERENCE=1) by default.
export CI=true
export _SKYRL_USE_NEW_INFERENCE=0

uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

uv run --directory . --isolated --extra dev --extra fsdp pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py::test_token_based_generation \
    tests/backends/skyrl_train/gpu/gpu_ci/test_save_weights_for_sampler.py::test_save_weights_for_sampler_then_inference \
    tests/backends/skyrl_train/gpu/gpu_ci/test_skyrl_gym_generator.py::test_generator_single_turn_gsm8k

uv run --directory . --isolated --extra dev --extra megatron pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_worker.py::test_megatron_policy_weight_sync
