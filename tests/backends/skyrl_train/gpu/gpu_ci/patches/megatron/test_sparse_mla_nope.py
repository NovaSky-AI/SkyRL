"""``patch_sparse_mla_nope`` (NVIDIA/Megatron-LM#7617 backport) against the real TileLang kernel.

The TileLang SparseMLA kernel is specialized for the DeepSeek-V3.2 absorbed layout (q/k width
576 = 512 latent + 64 RoPE, top-k width a multiple of 64). GLM-5.3-Flash is NoPE MLA (width 512)
with a 2048 + pool_size - 1 = 2051 wide k-pool selection, so unpatched the kernel declines and
megatron-core runs a dense ``[b, heads, sq, sk]`` masked softmax. The patch zero-pads q/k into the
RoPE slot and pads the indices with -1; these check the fused output and gradients against an
unpadded reference: upstream's small-shape sweep, and a GLM-5.3-Flash-shaped case against
megatron-core's own dense fallback.

H100-only: the SparseMLA kernel needs ~166 KB of shared memory per block, more than an L4
(sm89) allows ("Failed to set the allowed dynamic shared memory size to 169984").

Run with:
uv run --isolated --extra dev --extra megatron pytest -s -m h100 \
    tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/test_sparse_mla_nope.py
"""

import types

import pytest
import torch

pytestmark = [pytest.mark.megatron, pytest.mark.h100]

LATENT = 512  # GLM-5.3-Flash kv_lora_rank; NoPE, so q/k width == latent == value width
POOL_SIZE = 4
INDEX_TOPK = 2048


@pytest.fixture(scope="module")
def tilelang_dsa():
    pytest.importorskip("tilelang")
    from megatron.core.transformer.experimental_attention_variant.ops import (
        tilelang_dsa,
    )

    if tilelang_dsa.SparseMLA is None:
        pytest.skip("TileLang SparseMLA is unavailable")

    from skyrl.backends.skyrl_train.patches.megatron.patch_sparse_mla_nope import (
        patch_sparse_mla_nope,
    )

    assert patch_sparse_mla_nope() is True
    return tilelang_dsa


def _reference(query, key, indices, scale):
    """Sparse absorbed attention in FP32 on the unpadded inputs (upstream's reference)."""
    # Each row selects its own keys; the first 512 channels are also the values.
    query = query.permute(1, 0, 2, 3).float()
    key = key[:, :, 0].permute(1, 0, 2).float()
    batch = torch.arange(key.size(0), device=key.device)[:, None, None]
    selected = key[batch, indices.clamp_min(0).long()]
    valid = indices >= 0
    logits = torch.einsum("bshd,bskd->bshk", query, selected) * scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    # All-invalid rows must produce zero output and zero gradients.
    logits = torch.where(valid.any(-1)[:, :, None, None], logits, torch.zeros_like(logits))
    probs = logits.softmax(-1).masked_fill(~valid.unsqueeze(2), 0)
    return torch.einsum("bshk,bskd->bshd", probs, selected[..., :512]).permute(1, 0, 2, 3)


@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [64, 65, 127])
@pytest.mark.parametrize("heads", [8, 16, 32])
def test_absorbed_shapes_match_reference_on_cuda(tilelang_dsa, dim, topk, heads):
    """Compare real TileLang output and both input gradients to unpadded sparse attention."""
    torch.manual_seed(17)
    length = 129
    query = torch.randn(length, 1, heads, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(length, 1, 1, dim, device="cuda", dtype=torch.bfloat16)
    query.requires_grad_()
    key.requires_grad_()
    rows = torch.arange(length, device="cuda")[:, None]
    indices = (rows - torch.arange(topk, device="cuda")[None, :]).clamp_min(-1)
    indices[0] = -1
    indices = indices.unsqueeze(0).to(torch.int32)
    scale = 0.037
    output = tilelang_dsa.fused_sparse_mla_absorbed(query, key, indices, scale, 512)
    assert output is not None, "supported shapes must reach the fused kernel"
    grad = torch.randn_like(output) * 0.01
    output.backward(grad)
    q_ref = query.detach().clone().requires_grad_()
    k_ref = key.detach().clone().requires_grad_()
    reference = _reference(q_ref, k_ref, indices, scale)
    reference.backward(grad.float())
    for actual, expected in ((output, reference), (query.grad, q_ref.grad), (key.grad, k_ref.grad)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.05, atol=0.01)
        relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert relative_error < 0.02
    assert torch.count_nonzero(output[0]) == 0
    assert torch.count_nonzero(query.grad[0]) == 0


def _causal_kpool_like_indices(seqlen: int, width: int, device: str, seed: int = 0) -> torch.Tensor:
    """[1, seqlen, width] causal token indices, -1 in unused slots (as k-pool emits for short rows)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = torch.full((1, seqlen, width), -1, dtype=torch.int64, device=device)
    for s in range(seqlen):
        n = min(s + 1, width)
        idx[0, s, :n] = torch.randperm(s + 1, device=device, generator=gen)[:n]
    return idx


@pytest.mark.parametrize("seqlen", [1024, 4096])
def test_glm5_next_shape_through_dsa_hook_matches_dense_fallback(tilelang_dsa, seqlen):
    """GLM-5.3-Flash shapes (16 heads = 64 / TP4, 2051-wide k-pool selection) via ``dsa_kernels``.

    Goes through the same hook ``DSAttention`` calls and compares against the dense fallback it
    would otherwise run, so a decline here is a regression back to O(L^2) memory.
    """
    from megatron.core.transformer.experimental_attention_variant import (
        dsa as mcore_dsa,
    )
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels

    heads = 16
    torch.manual_seed(0)
    device = "cuda"
    q = (torch.randn(seqlen, 1, heads, LATENT, device=device) * 0.5).bfloat16().requires_grad_()
    k = (torch.randn(seqlen, 1, 1, LATENT, device=device) * 0.5).bfloat16().requires_grad_()
    idx = _causal_kpool_like_indices(seqlen, INDEX_TOPK + POOL_SIZE - 1, device)
    scale = LATENT**-0.5
    cfg = types.SimpleNamespace(dsa_kernel_backend="tilelang", attention_backend="auto")

    out = dsa_kernels.run_fused_absorbed_sparse_attention(cfg, q, k, idx, scale, LATENT)
    assert out is not None, "fused sparse attention declined GLM-5.3-Flash's layout"
    ref = mcore_dsa._unfused_absorbed_dsa_fn(q, k, idx, scale, LATENT)
    assert out.shape == ref.shape == (seqlen, 1, heads, LATENT)

    grad_out = torch.randn(ref.shape, device=device)
    dq, dk = torch.autograd.grad((out.float() * grad_out).sum(), (q, k))
    dq_ref, dk_ref = torch.autograd.grad((ref.float() * grad_out).sum(), (q, k))

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    # bf16 kernel vs fp32-softmax reference.
    assert rel(out, ref) < 1e-2
    assert rel(dq, dq_ref) < 1e-2
    assert rel(dk, dk_ref) < 1e-2
