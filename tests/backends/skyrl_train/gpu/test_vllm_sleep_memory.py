"""
Test that vLLM sleep mode properly frees model weights from GPU memory.

This validates the SkyRL monkey-patch for a vLLM v0.16.0 bug where
gpu_worker.load_model() uses `with A and B:` instead of `with A, B:`,
causing CuMemAllocator to never track model weights.

The test FAILS without the patch (model weights not freed) and
PASSES with it (model weights properly freed via CuMemAllocator).

Fixed upstream: https://github.com/vllm-project/vllm/pull/32947 (v0.17.0+)
Patched in SkyRL: skyrl/backends/skyrl_train/inference_servers/vllm_worker.py

Requires: 1 GPU (any type with >= 4 GiB)
Run: pytest tests/backends/skyrl_train/gpu/test_vllm_sleep_memory.py -v -s
"""

import gc
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")

TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _gpu_used_bytes() -> int:
    free, total = torch.cuda.mem_get_info()
    return total - free


def _create_engine_and_measure_sleep():
    """Create a vLLM engine with sleep mode, sleep it, return memory stats.

    Returns (model_bytes, cumem_freed_bytes, total_freed_bytes, residual_bytes).
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import vllm

    used_before = _gpu_used_bytes()

    engine = vllm.LLM(
        model=TEST_MODEL,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        enable_sleep_mode=True,
        max_model_len=512,
    )

    used_after_init = _gpu_used_bytes()
    model_and_cache = used_after_init - used_before

    # Capture CuMemAllocator stats via the sleep log
    # The allocator reports how much it freed — if weights are tracked,
    # this should include model weights + KV cache.
    # If NOT tracked (bug), this only includes KV cache.
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.get_instance()
    cumem_usage_before_sleep = allocator.get_current_usage()

    engine.sleep(level=2)
    gc.collect()
    torch.cuda.empty_cache()

    used_after_sleep = _gpu_used_bytes()
    total_freed = used_after_init - used_after_sleep
    residual = used_after_sleep - used_before

    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return model_and_cache, cumem_usage_before_sleep, total_freed, residual


def test_sleep_frees_model_weights():
    """CuMemAllocator must track model weights so sleep can free them.

    Without the patch: CuMemAllocator only tracks KV cache (~few GiB for
    small models), so `cumem_usage` is small and model weights (~1 GiB)
    remain in GPU after sleep.

    With the patch: CuMemAllocator tracks weights + KV cache, so
    `cumem_usage` includes model weights and sleep frees everything.

    We assert that the residual GPU memory after sleep is < 30% of the
    loaded model+cache size. This is GPU-agnostic since it uses ratios.
    """
    # Import the monkey-patch — this is what we're testing
    import skyrl.backends.skyrl_train.inference_servers.vllm_worker  # noqa: F401

    model_and_cache, cumem_usage, total_freed, residual = (
        _create_engine_and_measure_sleep()
    )

    freed_pct = total_freed / model_and_cache * 100 if model_and_cache > 0 else 0
    residual_pct = residual / model_and_cache * 100 if model_and_cache > 0 else 0

    print(f"\nModel+cache loaded: {model_and_cache / 1024**3:.2f} GiB")
    print(f"CuMemAllocator tracked: {cumem_usage / 1024**3:.2f} GiB")
    print(f"Total freed by sleep: {total_freed / 1024**3:.2f} GiB ({freed_pct:.0f}%)")
    print(f"Residual after sleep: {residual / 1024**3:.2f} GiB ({residual_pct:.0f}%)")

    # Key assertion: residual should be small (< 30% of loaded).
    # Without the fix, residual is ~50%+ (model weights stuck in GPU).
    # With the fix, residual is ~5% (just CUDA context overhead).
    assert residual_pct < 30, (
        f"Sleep left {residual_pct:.0f}% of model+cache in GPU "
        f"(residual={residual / 1024**3:.2f} GiB). "
        f"Model weights are likely NOT tracked by CuMemAllocator. "
        f"Ensure the vLLM sleep patch is applied."
    )
