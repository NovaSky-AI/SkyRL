"""Qualify ragged top-k cuDNN backward against dense autograd on SM90/SM100."""

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.patch_dsa_backward_topk_align import (
    patch_dsa_backward_topk_align,
)


@pytest.mark.megatron
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA, FlashMLA and cuDNN DSA")
@pytest.mark.parametrize("width", [65, 2051])
def test_ragged_topk_backward_matches_dense_attention(width):
    from megatron.core.transformer.experimental_attention_variant import (
        dsa_cudnn_kernels as kernels,
    )

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("DSA kernel qualification requires SM90 or newer")
    # megatron-core's cuDNN DSA forward runs on FlashMLA, which SkyRL's megatron extra does not ship.
    pytest.importorskip("flash_mla", reason="cuDNN DSA sparse attention forward requires FlashMLA")
    patch_dsa_backward_topk_align()
    torch.manual_seed(42)
    sq, b, heads, dim, skv = 4, 1, 8, 512, width + 61
    query = (0.1 * torch.randn(sq, b, heads, dim, device="cuda")).bfloat16()
    kv = (0.1 * torch.randn(skv, b, dim, device="cuda")).bfloat16()
    indices = torch.arange(width, device="cuda", dtype=torch.int32).expand(b, sq, width).contiguous()
    scale = dim**-0.5
    out, lse, q_flat, kv_flat, sink, global_idxs, lengths = kernels._run_sparse_attention_forward(
        query, kv, indices, scale, d_v=dim
    )
    assert global_idxs.shape[-1] == width
    grad = torch.randn_like(out)
    dq, dkv = kernels._run_sparse_attention_backward(
        q_flat=q_flat,
        kv_flat=kv_flat,
        attn_sink=sink,
        global_idxs=global_idxs,
        out_flat=out,
        lse=lse,
        topk_length=lengths,
        softmax_scale=scale,
        sq=sq,
        b=b,
        num_heads=heads,
        d=dim,
        skv=skv,
        grad_output=grad,
    )
    torch.cuda.synchronize()

    q_ref = query[:, 0].float().requires_grad_()
    kv_ref = kv[:, 0].float().requires_grad_()
    scores = torch.einsum("qhd,kd->qhk", q_ref, kv_ref[:width]) * scale
    expected = torch.einsum("qhk,kd->qhd", scores.softmax(-1), kv_ref[:width])
    expected.backward(grad.float())
    # The fused kernels operate on BF16 values; the oracle accumulates in FP32.
    torch.testing.assert_close(out.float(), expected, rtol=0.04, atol=0.003)
    torch.testing.assert_close(dq.reshape_as(q_ref).float(), q_ref.grad, rtol=0.04, atol=0.003)
    torch.testing.assert_close(dkv.reshape_as(kv_ref).float(), kv_ref.grad, rtol=0.04, atol=0.003)
    assert torch.isfinite(dq).all() and torch.isfinite(dkv).all()
