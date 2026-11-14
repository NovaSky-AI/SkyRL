#!/usr/bin/env bash
set -xuo pipefail

export CI=true

# Usage: ci/gpu_ci_run_h100.sh [models|other|all]
#
# The full H100 suite takes ~5.5 h on one 4xH100 node, longer than the 4.5 h run timeout in
# .github/workflows/gpu_ci_h100.yaml, so the workflow runs two shards as parallel jobs:
#   models -- test_megatron_models.py (~3 h)
#   other  -- FSDP + the remaining Megatron files (~2.5 h)
# `all` (the default) runs both, in order, in one job.
shard="${1:-all}"
case "$shard" in
    models | other | all) ;;
    *)
        echo "Unknown H100 CI shard: $shard" >&2
        exit 2
        ;;
esac

# Prepare datasets used in tests (Megatron test uses gsm8k env_class).
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k || exit $?

test_root=tests/backends/skyrl_train/gpu/gpu_ci

# Run every suite in the shard even if an earlier one fails, so one failure doesn't hide
# the rest of the shard's results; exit non-zero if any of them failed.
status=0
run() {
    "$@" || status=$?
}

if [[ "$shard" == models || "$shard" == all ]]; then
    run uv run --directory . --isolated --extra dev --extra megatron pytest -s -vvv -m h100 \
        "$test_root/megatron/test_megatron_models.py"
fi

if [[ "$shard" == other || "$shard" == all ]]; then
    # FSDP h100 tests.
    run uv run --directory . --isolated --extra dev --extra fsdp pytest -s -vvv -m h100 \
        "$test_root/test_policy_local_engines_e2e.py"
    # Remaining Megatron h100 tests.
    run uv run --directory . --isolated --extra dev --extra megatron pytest -s -vvv -m h100 \
        "$test_root/megatron/test_router_replay.py" \
        "$test_root/megatron/test_kimi_k25_bridge.py" \
        "$test_root/megatron/test_megatron_lora_models.py" \
        "$test_root/megatron/test_megatron_vlm_init.py" \
        "$test_root/megatron/test_qwen38_qsa_long_context.py" \
        "$test_root/patches/megatron/test_sparse_mla_nope.py" \
        "$test_root/patches/megatron/test_triton_kernels_int64_offsets.py"
fi

exit "$status"
