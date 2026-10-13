"""CPU test for ``patch_sparse_mla_nope`` (NVIDIA/Megatron-LM#7617 backport).

Ported from upstream's ``test_absorbed_shape_adapter_preserves_inputs_and_gradients``: a fake
``SparseMLA`` checks the padding, sentinels, scale and unpadding the patched
``fused_sparse_mla_absorbed`` hands the kernel, without needing TileLang or a GPU. The real-kernel
half is ``tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/test_sparse_mla_nope.py``.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")

# Runs in the CPU megatron job (`-m megatron`); without the marker that job deselects it.
pytestmark = pytest.mark.megatron


@pytest.fixture
def tilelang_dsa():
    from megatron.core.transformer.experimental_attention_variant.ops import (
        tilelang_dsa,
    )

    from skyrl.backends.skyrl_train.patches.megatron.patch_sparse_mla_nope import (
        patch_sparse_mla_nope,
    )

    assert patch_sparse_mla_nope() is True
    return tilelang_dsa


def test_patch_is_idempotent(tilelang_dsa):
    from skyrl.backends.skyrl_train.patches.megatron.patch_sparse_mla_nope import (
        _fused_sparse_mla_absorbed,
        patch_sparse_mla_nope,
    )

    assert patch_sparse_mla_nope() is True
    assert tilelang_dsa.fused_sparse_mla_absorbed is _fused_sparse_mla_absorbed


@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [64, 65, 127, 2051])
@pytest.mark.parametrize("heads", [8, 16, 32])
def test_absorbed_shape_adapter_preserves_inputs_and_gradients(tilelang_dsa, monkeypatch, dim, topk, heads):
    """Check padding, sentinels, scale and unpadding without requiring the optional kernel."""
    query = torch.randn(3, 2, heads, dim, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(5, 2, 1, dim, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.full((2, 3, topk), -1, dtype=torch.int32)
    indices[..., 0] = 0
    originals = [tensor.detach().clone() for tensor in (query, key, indices)]
    scale = 0.037

    def fake_sparse_mla(q, k, slots, softmax_scale):
        assert softmax_scale == scale
        assert q.shape == (2, 3, max(heads, 16), 576)
        assert k.shape == (2, 5, 1, 576)
        assert slots.shape == (2, 3, 1, ((topk + 63) // 64) * 64)
        torch.testing.assert_close(slots[:, :, 0, :topk], indices)
        assert (slots[..., topk:] == -1).all()
        if dim == 512:
            assert torch.count_nonzero(q[..., 512:]) == 0
            assert torch.count_nonzero(k[..., 512:]) == 0
        if heads < 16:
            assert torch.count_nonzero(q[:, :, heads:]) == 0
        return q[..., :512] + k[:, :1, :, :512], None

    monkeypatch.setattr(tilelang_dsa, "SparseMLA", SimpleNamespace(apply=fake_sparse_mla))
    # Through the hook megatron-core's DSAttention calls, not the patched function directly.
    output = tilelang_dsa.run_fused_absorbed_sparse_attention(query, key, indices, scale, 512)
    assert output is not None
    assert output.shape == (3, 2, heads, 512)
    output.sum().backward()
    expected_q = torch.zeros_like(query)
    expected_q[..., :512] = 1
    expected_k = torch.zeros_like(key)
    expected_k[0, ..., :512] = 3 * heads
    torch.testing.assert_close(query.grad, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(key.grad, expected_k, atol=0, rtol=0)
    for tensor, original in zip((query, key, indices), originals):
        torch.testing.assert_close(tensor, original, atol=0, rtol=0)
