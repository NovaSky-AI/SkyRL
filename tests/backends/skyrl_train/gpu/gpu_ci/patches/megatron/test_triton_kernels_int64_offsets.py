"""``patch_triton_kernels_int64_offsets.py``: megatron-core Triton kernels past their int32 row-offset limit.

- fused mHC (``n = 4`` streams of ``C = 5120``, DeepSeek-V4 / V4.1): int32 offsets wrap after ~104.9k rows. Run
  the Triton backend of ``fused_h_aggregate`` and ``fused_h_post_bda`` (forward and backward, all four
  row-indexed kernels) on 110k rows and compare the last rows -- past the wrap -- with the row-local native
  implementations on those rows alone.
- MLA RoPE in place (64 heads of 448 + 64 dims): wraps after 65,536 rows. Run it on 70k rows (forward and
  backward) and compare the tail with the same kernel on the tail alone (cos / sin rows sliced to match).
- MLA RoPE KV split (128 heads of 128 + 128 dims): wraps after 65,536 rows. Same check on 70k rows.

Each case keeps several >2**31-element tensors (and their grads) alive, ~20 GB, so these run in the opt-in H100
suite (ci/gpu_ci_run_h100.sh), not on the L4 megatron runners.
"""

import pytest
import torch

pytestmark = [pytest.mark.megatron, pytest.mark.h100]

ROWS, STREAMS, CHANNELS, TAIL = 110_000, 4, 5120, 512


def test_mhc_kernels_past_int32_row_offsets():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from megatron.core.fusions import fused_mhc_kernels as mhc

    from skyrl.backends.skyrl_train.patches.megatron.patch_triton_kernels_int64_offsets import (
        apply_triton_kernels_int64_offsets_patch,
    )

    if not getattr(mhc, "HAVE_TRITON", True):
        pytest.skip("needs Triton")
    assert ROWS * STREAMS * CHANNELS > 2**31
    assert apply_triton_kernels_int64_offsets_patch()["megatron.core.fusions.fused_mhc_kernels"]
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


def test_mla_rope_inplace_past_int32_row_offsets():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from megatron.core.fusions import fused_mla_yarn_rope_apply as rope

    from skyrl.backends.skyrl_train.patches.megatron.patch_triton_kernels_int64_offsets import (
        apply_triton_kernels_int64_offsets_patch,
    )

    rows, heads, nope, emb, tail_rows = 70_000, 64, 448, 64, 512
    assert rows * heads * (nope + emb) > 2**31
    assert apply_triton_kernels_int64_offsets_patch()["megatron.core.fusions.fused_mla_yarn_rope_apply"]
    torch.manual_seed(0)
    angles = torch.rand(rows, 1, 1, emb, device="cuda") * 6.28
    cos, sin = angles.cos(), angles.sin()
    tail = slice(rows - tail_rows, rows)

    leaf = torch.randn(rows, 1, heads, nope + emb, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = rope.fused_mla_rope_inplace(leaf.clone(), cos, sin, nope, emb)
    grad = torch.zeros_like(out)
    grad[tail] = torch.randn_like(out[tail])
    out.backward(grad)

    ref_leaf = leaf[tail].detach().clone().requires_grad_()
    ref = rope.fused_mla_rope_inplace(ref_leaf.clone(), cos[tail].contiguous(), sin[tail].contiguous(), nope, emb)
    ref.backward(grad[tail])
    torch.testing.assert_close(out[tail], ref)
    torch.testing.assert_close(leaf.grad[tail], ref_leaf.grad)


def test_mla_rope_kv_split_past_int32_row_offsets():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from megatron.core.fusions import fused_mla_yarn_rope_apply as rope

    from skyrl.backends.skyrl_train.patches.megatron.patch_triton_kernels_int64_offsets import (
        apply_triton_kernels_int64_offsets_patch,
    )

    rows, heads, k_dim, v_dim, emb, tail_rows = 70_000, 128, 128, 128, 64, 512
    assert rows * heads * (k_dim + v_dim) > 2**31
    assert apply_triton_kernels_int64_offsets_patch()["megatron.core.fusions.fused_mla_yarn_rope_apply"]
    torch.manual_seed(0)
    angles = torch.rand(rows, 1, 1, emb, device="cuda") * 6.28
    cos, sin = angles.cos(), angles.sin()
    tail = slice(rows - tail_rows, rows)

    kv = torch.randn(rows, 1, heads, k_dim + v_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k_pos = torch.randn(rows, 1, 1, emb, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key, value = rope.fused_mla_rope_kv_split(kv, k_pos, cos, sin, emb, k_dim, v_dim)
    g_key, g_value = torch.zeros_like(key), torch.zeros_like(value)
    g_key[tail], g_value[tail] = torch.randn_like(key[tail]), torch.randn_like(value[tail])
    torch.autograd.backward((key, value), (g_key, g_value))

    kv_ref = kv[tail].detach().clone().requires_grad_()
    k_pos_ref = k_pos[tail].detach().clone().requires_grad_()
    key_ref, value_ref = rope.fused_mla_rope_kv_split(
        kv_ref, k_pos_ref, cos[tail].contiguous(), sin[tail].contiguous(), emb, k_dim, v_dim
    )
    torch.autograd.backward((key_ref, value_ref), (g_key[tail], g_value[tail]))
    torch.testing.assert_close(key[tail], key_ref)
    torch.testing.assert_close(value[tail], value_ref)
    torch.testing.assert_close(kv.grad[tail], kv_ref.grad)
    torch.testing.assert_close(k_pos.grad[tail], k_pos_ref.grad)
