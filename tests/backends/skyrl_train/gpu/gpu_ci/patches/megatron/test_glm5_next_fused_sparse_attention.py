"""GLM-5.3-Flash query-chunked fused sparse attention, on GPU.

``SKYRL_DSA_QUERY_CHUNK`` makes ``glm5_next/dsa.py`` run the fused TileLang SparseMLA kernel in
checkpointed query chunks. The kernel takes GLM-5.3-Flash's NoPE (width 512), 2051-wide k-pool
layout through ``patch_sparse_mla_nope`` (its own test is ``test_sparse_mla_nope.py``); this
checks that chunking reproduces the one-shot kernel, forward and backward.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/test_glm5_next_fused_sparse_attention.py
"""

import types

import pytest
import torch

pytestmark = pytest.mark.megatron

LATENT = 512  # GLM-5.3-Flash kv_lora_rank; NoPE, so q/k width == latent == value width
HEADS = 16  # 64 heads / TP4
POOL_SIZE = 4
INDEX_TOPK = 2048


def _causal_kpool_like_indices(seqlen: int, width: int, device: str, seed: int = 0) -> torch.Tensor:
    """[1, seqlen, width] causal token indices, -1 in unused slots (as k-pool emits for short rows)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = torch.full((1, seqlen, width), -1, dtype=torch.int64, device=device)
    for s in range(seqlen):
        n = min(s + 1, width)
        idx[0, s, :n] = torch.randperm(s + 1, device=device, generator=gen)[:n]
    return idx


def test_query_chunked_fused_sparse_attention_matches_unchunked(monkeypatch):
    """``SKYRL_DSA_QUERY_CHUNK``: checkpointed query chunks must reproduce the one-shot kernel.

    Each query attends only its own top-k keys, so chunking over queries is exact up to the
    summation order of the key gradient across chunks. 1000 does not divide the sequence, so the
    last chunk is ragged.
    """
    pytest.importorskip("tilelang")
    from megatron.core.transformer.experimental_attention_variant import (
        dsa as mcore_dsa,
    )
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels

    from skyrl.backends.skyrl_train.patches.megatron.glm5_next import dsa as glm_dsa
    from skyrl.backends.skyrl_train.patches.megatron.patch_sparse_mla_nope import (
        patch_sparse_mla_nope,
    )

    assert patch_sparse_mla_nope() is True
    seqlen = 4096
    torch.manual_seed(0)
    device = "cuda"
    q = (torch.randn(seqlen, 1, HEADS, LATENT, device=device) * 0.5).bfloat16().requires_grad_()
    k = (torch.randn(seqlen, 1, 1, LATENT, device=device) * 0.5).bfloat16().requires_grad_()
    idx = _causal_kpool_like_indices(seqlen, INDEX_TOPK + POOL_SIZE - 1, device)
    scale = LATENT**-0.5
    cfg = types.SimpleNamespace(dsa_kernel_backend="tilelang", attention_backend="auto")
    fused = glm_dsa._query_chunked_fused_absorbed_sparse_attention(dsa_kernels.run_fused_absorbed_sparse_attention)
    grad_out = torch.randn(seqlen, 1, HEADS, LATENT, device=device)

    def run():
        out = fused(cfg, q, k, idx, scale, LATENT)
        assert out is not None
        dq, dk = torch.autograd.grad((out.float() * grad_out).sum(), (q, k))
        return out, dq, dk

    monkeypatch.setattr(glm_dsa, "_DSA_QUERY_CHUNK", 0)
    out, dq, dk = run()
    monkeypatch.setattr(glm_dsa, "_DSA_QUERY_CHUNK", 1000)
    out_c, dq_c, dk_c = run()
    ref = mcore_dsa._unfused_absorbed_dsa_fn(q, k, idx, scale, LATENT)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    assert rel(out_c, out) < 1e-3
    assert rel(dq_c, dq) < 1e-3
    assert rel(dk_c, dk) < 1e-2  # key grads are summed across chunks in a different order
    assert rel(out_c, ref) < 1e-2
