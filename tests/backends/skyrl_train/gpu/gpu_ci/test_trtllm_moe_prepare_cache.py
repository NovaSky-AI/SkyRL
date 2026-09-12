"""The TRTLLM MoE prepare cache must be bitwise-invisible.

It replaces vLLM's per-expert shuffle loop (re-run on every weight sync by
``process_weights_after_loading``) with learned row/chunk permutations; every
output must equal the original's bitwise, changed values must flow through,
and unlearnable transforms must fall back rather than corrupt.
"""

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    is_blackwell_or_newer,
)

pytestmark = pytest.mark.skipif(
    not is_blackwell_or_newer(),
    reason="requires SM100+: the cache wraps the TRT-LLM MXFP8 prepare path, " "which only engages on Blackwell",
)


def _shuffle_module():
    import vllm.model_executor.layers.fused_moe  # noqa: F401  (break circular import)
    from vllm.model_executor.layers.quantization.utils import flashinfer_utils

    return flashinfer_utils


def _cache_module():
    from skyrl.backends.skyrl_train.inference_servers import trtllm_moe_prepare_cache

    return trtllm_moe_prepare_cache


def _make(E, N2, K, dev="cuda"):
    torch.manual_seed(0)
    w13 = torch.randint(0, 256, (E, N2, K), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    w2 = torch.randint(0, 256, (E, K, N2 // 4), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    w13_scale = torch.randint(0, 256, (E, N2, K // 32), dtype=torch.uint8, device=dev)
    w2_scale = torch.randint(0, 256, (E, K, N2 // 4 // 32), dtype=torch.uint8, device=dev)
    return w13, w2, w13_scale, w2_scale


@pytest.fixture()
def patched():
    cache = _cache_module()
    fi = _shuffle_module()
    original = fi._shuffle_mxfp8_moe_weights
    cache.uninstall()
    cache._CACHE.clear()
    assert cache.install()
    yield cache, fi, original
    cache.uninstall()
    cache._CACHE.clear()
    assert fi._shuffle_mxfp8_moe_weights is original


def test_cached_prepare_is_bitwise_identical(patched):
    cache, fi, original = patched
    tensors = _make(8, 512, 512)
    ref = original(*tensors, True)
    learn = fi._shuffle_mxfp8_moe_weights(*tensors, True)
    hit = fi._shuffle_mxfp8_moe_weights(*tensors, True)
    for r, a, b in zip(ref, learn, hit):
        assert torch.equal(r.view(torch.uint8), a.view(torch.uint8))
        assert torch.equal(r.view(torch.uint8), b.view(torch.uint8))
        assert r.shape == b.shape and r.dtype == b.dtype
    assert any(entry is not None for entry in cache._CACHE.values()), "expected a learned entry, not fallback"


def test_cached_prepare_tracks_changed_values(patched):
    cache, fi, original = patched
    tensors = _make(4, 512, 512)
    fi._shuffle_mxfp8_moe_weights(*tensors, True)  # learn
    w13b = (tensors[0].view(torch.uint8) + 17).view(torch.float8_e4m3fn)
    changed = (w13b, *tensors[1:])
    got = fi._shuffle_mxfp8_moe_weights(*changed, True)
    ref = original(*changed, True)
    for r, g in zip(ref, got):
        assert torch.equal(r.view(torch.uint8), g.view(torch.uint8))


def test_unlearnable_transform_falls_back():
    """A transform that is not a pure relocation must be detected and bypassed."""

    cache = _cache_module()
    fi = _shuffle_module()
    true_original = fi._shuffle_mxfp8_moe_weights

    def not_a_permutation(w13, w2, w13_scale, w2_scale, is_gated):
        out = true_original(w13, w2, w13_scale, w2_scale, is_gated)
        w13_out = (out[0].view(torch.uint8) + 1).view(torch.float8_e4m3fn)  # arithmetic: not a relocation
        return (w13_out, *out[1:])

    cache.uninstall()
    cache._CACHE.clear()
    fi._shuffle_mxfp8_moe_weights = not_a_permutation
    try:
        assert cache.install()
        tensors = _make(4, 512, 512)
        got = fi._shuffle_mxfp8_moe_weights(*tensors, True)
        ref = not_a_permutation(*tensors, True)
        for r, g in zip(ref, got):
            assert torch.equal(r.view(torch.uint8), g.view(torch.uint8))
        assert all(entry is None for entry in cache._CACHE.values()), "expected validated fallback"
    finally:
        cache.uninstall()
        cache._CACHE.clear()
        fi._shuffle_mxfp8_moe_weights = true_original
