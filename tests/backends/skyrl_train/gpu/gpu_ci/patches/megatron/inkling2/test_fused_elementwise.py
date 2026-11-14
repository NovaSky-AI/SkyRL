"""Inkling2's compiled elementwise pieces (``modules.fused``) match their eager versions, values and gradients.

The compiled kernels compute in FP32 like the eager ops. They differ by reassociation and by skipping some intermediate
BF16 roundings of the eager code (Inductor drops a BF16 round trip inside a fused kernel), so BF16 outputs agree to
about one BF16 ulp and gradients to well under 1% of their norm. Two lengths per function: the compiled code is dynamic
in the sequence length, and the second call must not recompile into something different.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/inkling2/test_fused_elementwise.py
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from skyrl.backends.skyrl_train.patches.megatron.inkling2 import modules  # noqa: E402

pytestmark = pytest.mark.megatron

H, HEADS, KV, D, D_REL, W = 256, 4, 2, 64, 8, 4


def _leaf(*shape, dtype=torch.bfloat16, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).to(dtype).requires_grad_()


def _check(fn, make_inputs, seed=0):
    """Compare ``fn`` with ``fn.eager`` on fresh copies of the same inputs: outputs and input gradients."""
    for s in (37, 96):
        torch.manual_seed(seed + s)
        inputs = make_inputs(s)
        copies = [x.detach().clone().requires_grad_(x.requires_grad) if torch.is_tensor(x) else x for x in inputs]
        outs = [fn(*inputs), fn.eager(*copies)]
        flat = [o if isinstance(o, (tuple, list)) else (o,) for o in outs]
        torch.manual_seed(1)
        grads = [torch.randn_like(o, dtype=torch.float32) for o in flat[0]]
        for got, ref in zip(*flat):
            tol = 1e-5 if got.dtype == torch.float32 else 1e-2
            torch.testing.assert_close(got.float(), ref.float(), rtol=tol, atol=tol)
        for o in flat:
            torch.autograd.backward([t.float() for t in o], grads)
        for x, y in zip(inputs, copies):
            if torch.is_tensor(x) and x.requires_grad:
                # Relative to the gradient's norm: eager rounds some intermediate gradients to BF16 (the gradient of
                # a BF16 intermediate) where the fused backward keeps FP32, and a gradient reduced over tokens (the
                # per-head scales) sums those per-token roundings. A layout or formula bug is an O(1) error.
                # Some gradients are exactly zero (softmax over one source, halo rows the positions mask out).
                diff, ref = (x.grad.float() - y.grad.float()).norm().item(), y.grad.float().norm().item()
                assert (
                    diff <= 1e-2 * ref + 1e-5
                ), f"gradient of a {tuple(x.shape)} input: |diff| {diff:.3g}, |ref| {ref:.3g}"


def test_rms_norm():
    _check(modules.rms_norm, lambda s: (_leaf(s, 1, H, dtype=torch.float32), _leaf(H), 1e-6, torch.bfloat16))
    _check(modules.rms_norm, lambda s: (_leaf(s, 1, KV, D), None, 1e-6, torch.bfloat16))


@pytest.mark.parametrize("n_sources", [1, 4])
@pytest.mark.parametrize("addend", [False, True])
@pytest.mark.parametrize("n_queries", [1, 3])
def test_block_attn_res(n_sources, addend, n_queries):
    def make(s):
        sources = [_leaf(s, 1, H) for _ in range(n_sources)]
        queries = _leaf(H, dtype=torch.float32) if n_queries == 1 else _leaf(n_queries, H, dtype=torch.float32)
        return sources, _leaf(s, 1, H) if addend else None, queries, True

    # Sources are a list: compare through a wrapper that takes the list's tensors as separate leaves.
    def wrap(fn):
        def call(*args):
            *sources, addend, queries = args
            return fn(list(sources), addend, queries, True)

        return call

    fused, eager = wrap(modules.block_attn_res), wrap(modules.block_attn_res.eager)
    fused.eager = eager

    def make_flat(s):
        sources, add, queries, _ = make(s)
        return (*sources, add, queries)

    _check(fused, make_flat)


def test_l2_clip_and_attention_inputs():
    _check(modules.l2_clip, lambda s: (_leaf(s, 1, HEADS, D, scale=30.0), 22.6))

    def make(s):
        tau = 1.0 + torch.rand(s, 1, device="cuda")
        return (
            _leaf(s, 1, HEADS, D, scale=10.0),
            _leaf(s, 1, KV, D, scale=30.0),
            _leaf(s, 1, KV, D),
            _leaf(s, 1, HEADS, D_REL, scale=5.0),
            _leaf(HEADS, D),
            _leaf(HEADS, D_REL),
            tau,
            22.6,
            8.0,
            1e-6,
        )

    _check(modules.attention_inputs, make)


def test_short_convs():
    def positions(s):
        # Two packed sequences, the second starting mid-slice.
        p = torch.arange(s, device="cuda")
        return torch.where(p < s // 3, p, p - s // 3).view(s, 1)

    _check(
        modules.short_conv_static,
        lambda s: (_leaf(s, 1, H), _leaf(H, 1, W, scale=0.5), positions(s), _leaf(W - 1, 1, H, dtype=torch.float32)),
    )
    _check(modules.short_conv_static, lambda s: (_leaf(s, 1, H), _leaf(H, 1, W, scale=0.5), positions(s), None))
    _check(
        modules.short_conv_dynamic,
        lambda s: (
            _leaf(s, 1, KV * D),
            _leaf(KV * D, W, scale=0.5),
            _leaf(s, 1, KV * W, scale=0.5),
            positions(s),
            None,
        ),
    )


def test_gated_attention_output_and_softcap():
    _check(
        modules.gated_attention_output,
        lambda s: (
            _leaf(1, HEADS, s, D, dtype=torch.float32),
            _leaf(1, HEADS, s, dtype=torch.float32, scale=3.0),
            _leaf(s, 1, HEADS * D),
            _leaf(s, 1, HEADS),
            torch.bfloat16,
        ),
    )
    _check(modules.softcap_logits, lambda s: (_leaf(s, 1, 512, scale=40.0), 30.0, 256, 500))


@pytest.mark.parametrize("boundary", [False, True])
def test_next_attn_res_state(boundary):
    def wrap(fn):
        def call(b0, b1, partial, attn, mlp):
            return fn([b0, b1], partial, attn, mlp, boundary)

        return call

    fused, eager = wrap(modules.next_attn_res_state), wrap(modules.next_attn_res_state.eager)
    fused.eager = eager
    _check(fused, lambda s: tuple(_leaf(s, 1, H) for _ in range(5)))
