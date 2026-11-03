"""``patch_mhc_kernels_int64_offsets.py``: fused mHC kernels past 2**31 / (n * C) rows.

With ``n = 4`` streams of ``C = 5120`` channels (DeepSeek-V4 / V4.1), int32 row offsets wrap after ~104.9k
rows. Run the Triton backend of ``fused_h_aggregate`` and ``fused_h_post_bda`` (forward and backward, all
four row-indexed kernels) on 110k rows and compare the last rows -- the ones past the wrap -- with the
row-local native implementations on those rows alone.
"""

import pytest
import torch

pytestmark = pytest.mark.megatron

ROWS, STREAMS, CHANNELS, TAIL = 110_000, 4, 5120, 512


def test_mhc_kernels_past_int32_row_offsets():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from megatron.core.fusions import fused_mhc_kernels as mhc

    from skyrl.backends.skyrl_train.patches.megatron.patch_mhc_kernels_int64_offsets import (
        apply_mhc_kernels_int64_offsets_patch,
    )

    if not getattr(mhc, "HAVE_TRITON", True):
        pytest.skip("needs Triton")
    assert ROWS * STREAMS * CHANNELS > 2**31
    assert apply_mhc_kernels_int64_offsets_patch()
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16

    def leaf(*shape, dtype=dt):
        return torch.randn(*shape, device=dev, dtype=dtype).requires_grad_()

    tail = slice(ROWS - TAIL, ROWS)

    # H aggregate: out[s] = sum_i h_pre[s, i] * x[s, i].
    x, h_pre = leaf(ROWS, 1, STREAMS, CHANNELS), leaf(ROWS, 1, STREAMS, dtype=torch.float32)
    out = mhc.fused_h_aggregate(x, h_pre, backend="triton")
    grad = torch.zeros_like(out)
    grad[tail] = torch.randn_like(out[tail])
    out.backward(grad)
    x_ref, h_ref = x[tail].detach().clone().requires_grad_(), h_pre[tail].detach().clone().requires_grad_()
    ref = mhc.native_h_aggregate(x_ref, h_ref)
    ref.backward(grad[tail])
    torch.testing.assert_close(out[tail].float(), ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x.grad[tail].float(), x_ref.grad.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(h_pre.grad[tail], h_ref.grad, rtol=2e-2, atol=5e-1)
    del x, h_pre, out, grad

    # H_post BDA: h_res.T @ residual + h_post * (x + bias).
    h_res = leaf(ROWS, 1, STREAMS, STREAMS, dtype=torch.float32)
    residual = leaf(ROWS, 1, STREAMS, CHANNELS)
    h_post = leaf(ROWS, 1, STREAMS, dtype=torch.float32)
    x = leaf(ROWS, 1, CHANNELS)
    out = mhc.fused_h_post_bda(h_res, residual, h_post, x, None, backend="triton")
    grad = torch.zeros_like(out)
    grad[tail] = torch.randn_like(out[tail])
    out.backward(grad)
    # The native reference in fp32 throughout (its bmm needs one dtype; the fused path mixes fp32 / bf16).
    refs = [t[tail].detach().float().requires_grad_() for t in (h_res, residual, h_post, x)]
    ref = mhc.native_h_post_bda(*refs, None)
    ref.backward(grad[tail])
    torch.testing.assert_close(out[tail].float(), ref.float(), rtol=2e-2, atol=5e-2)
    for got, want in zip((h_res, residual, h_post, x), refs):
        torch.testing.assert_close(got.grad[tail].float(), want.grad.float(), rtol=2e-2, atol=5e-1)
