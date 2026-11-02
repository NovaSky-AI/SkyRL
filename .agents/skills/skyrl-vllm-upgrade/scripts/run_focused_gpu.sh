#!/usr/bin/env bash
# Focused GPU validation for a vLLM bump: the inference / weight-sync / spec-decode
# slice of GPU CI, without the hours-long training suites.
#
# Usage (as an Anyscale entrypoint, from the repo root):
#   bash .agents/skills/skyrl-vllm-upgrade/scripts/run_focused_gpu.sh <group>...
#
# Groups (fsdp image): cpu_vllm  inference  weight_sync  mooncake
# Groups (megatron image): megatron
#
# Every group runs even if an earlier one fails; the exit code is non-zero if any
# failed, and a per-group summary is printed at the end (grep for "== SUMMARY").
set -uo pipefail

export CI=true
# Tests call bare ray.init(); keep them on a local instance, never a live cluster.
export RAY_ADDRESS=local

GPU_CI=tests/backends/skyrl_train/gpu/gpu_ci
declare -A RESULT

run_group() {
    local name=$1
    shift
    echo "================ GROUP $name: $*"
    local start=$SECONDS
    "$@"
    local rc=$?
    RESULT[$name]="rc=$rc ($((SECONDS - start))s)"
    echo "================ GROUP $name finished: ${RESULT[$name]}"
}

prepare_data() {
    uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
}

for group in "$@"; do
    case "$group" in
    cpu_vllm)
        # The CPU `vllm`-marker half of cpu_skyrl.yaml. It needs a Linux box with the vLLM
        # wheel; GPUs are hidden so it behaves as it does on CI's CPU runner.
        run_group cpu_vllm env CUDA_VISIBLE_DEVICES="" \
            uv run --isolated --extra fsdp --extra dev pytest -q tests/train/ tests/backends/skyrl_train/ \
            --ignore=tests/backends/skyrl_train/gpu -m "vllm"
        ;;
    inference)
        prepare_data
        # Engine startup, generation, HTTP API surface, LoRA serving, EP, vLLM runtime patches.
        run_group inference uv run --isolated --extra dev --extra fsdp pytest -s \
            "$GPU_CI/inference_servers" \
            "$GPU_CI/test_engine_generation.py" \
            "$GPU_CI/test_expert_parallel_inference.py" \
            "$GPU_CI/test_lora.py" \
            "$GPU_CI/patches/vllm" \
            -m "not (integrations or megatron or mooncake)"
        ;;
    weight_sync)
        prepare_data
        # Every transfer backend (nccl / ipc / delta / sharded_rdt) end to end, plus the
        # sleep/wake + KV offload path and prefix-cache reset that ride on it.
        run_group weight_sync uv run --isolated --extra dev --extra fsdp pytest -s \
            "$GPU_CI/test_policy_local_engines_e2e.py" \
            "$GPU_CI/test_offload_kv_weight_sync.py" \
            "$GPU_CI/test_prefix_cache_reset.py" \
            "$GPU_CI/test_delta_weight_sync_e2e.py" \
            "$GPU_CI/test_packed_nccl_mixed_dtype.py" \
            "$GPU_CI/test_save_weights_for_sampler.py" \
            -m "not (integrations or megatron or mooncake)"
        ;;
    mooncake)
        run_group mooncake uv run --isolated --extra dev --extra fsdp --extra mooncake pytest -s \
            "$GPU_CI" -m "mooncake" \
            --ignore="$GPU_CI/megatron" --ignore="$GPU_CI/patches/megatron"
        ;;
    megatron)
        prepare_data
        # Megatron -> vLLM weight sync on every backend (incl. sharded_rdt TP and PP x
        # multi-engine), MTP speculative decoding surviving a sync, the weight-source
        # channel agreement, and LoRA in-memory sync.
        run_group megatron uv run --isolated --extra dev --extra megatron pytest -s \
            "$GPU_CI/megatron/test_megatron_worker.py::test_megatron_policy_weight_sync" \
            "$GPU_CI/megatron/test_mtp_weight_sync.py" \
            "$GPU_CI/megatron/test_mtp_weight_roundtrip.py" \
            "$GPU_CI/megatron/test_megatron_weight_source.py" \
            "$GPU_CI/megatron/test_shared_outer_lora.py" \
            "$GPU_CI/test_lora.py" \
            -m "megatron"
        ;;
    *)
        echo "unknown group: $group" >&2
        RESULT[$group]="unknown group"
        ;;
    esac
done

echo "== SUMMARY"
failed=0
for g in "$@"; do
    echo "$g: ${RESULT[$g]}"
    [[ "${RESULT[$g]}" == rc=0* ]] || failed=1
done
exit $failed
