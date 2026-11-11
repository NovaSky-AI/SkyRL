#!/usr/bin/env bash
set -xeuo pipefail

export CI=true

# Prepare datasets used in tests (Megatron test uses gsm8k env_class).
uv run --isolated --extra megatron examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"

test_root=tests/backends/skyrl_train/gpu/gpu_ci
run_models() {
    uv run --isolated --extra dev --extra megatron pytest -s -vvv -m h100 \
        "$test_root/megatron/test_megatron_models.py"
}

run_other() {
    uv run --isolated --extra dev --extra fsdp pytest -s -vvv -m h100 \
        "$test_root/test_policy_local_engines_e2e.py"
    uv run --isolated --extra dev --extra megatron pytest -s -vvv -m h100 \
        "$test_root/megatron/test_router_replay.py" \
        "$test_root/megatron/test_kimi_k25_bridge.py" \
        "$test_root/megatron/test_megatron_lora_models.py" \
        "$test_root/megatron/test_megatron_vlm_init.py" \
        "$test_root/megatron/test_qwen38_qsa_long_context.py" \
        "$test_root/patches/megatron/test_sparse_mla_nope.py" \
        "$test_root/patches/megatron/test_triton_kernels_int64_offsets.py"
}

case "${1:-all}" in
    models) run_models ;;
    other) run_other ;;
    all)
        run_models
        run_other
        ;;
    *)
        echo "Unknown H100 CI shard: $1" >&2
        exit 2
        ;;
esac
