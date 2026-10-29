"""The mHC projection matches plain autograd bit for bit, without saving the FP32 upcast.

``RMSNormInputHyperConnectionModule._projection_and_get_norm`` runs the mapping projection and
RMS factor in FP32 on the bf16 residual stream. Plain autograd keeps the FP32 upcast of the input
for backward (2 GiB per mHC site at 32k tokens per rank on GLM-5.3-Flash); the module checkpoints
the upcast together with the math, so only the bf16 input is saved. Checked at several activation
scales, with both outputs or only one of them feeding the loss.

Pure tensor math, so it runs on CPU and lives outside ``gpu/``; the module-level comparison
against HF is ``gpu_ci/patches/megatron/mcore_ext/test_modules_vs_hf.py``.
"""

import types

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")

# Runs in the CPU megatron job (`-m megatron`); without the marker that job deselects it.
pytestmark = pytest.mark.megatron

S, B, WIDTH, OUT = 256, 2, 4 * 256, 24  # n=4 streams; OUT = n^2 + 2n mapping columns
EPS = 1e-5


def _plain(x, w):
    """Plain autograd: upcast, project, standard RMS factor."""
    s, b, nC = x.shape
    x32 = x.reshape(s * b, nC).to(torch.float32)
    proj = torch.matmul(x32, w.to(torch.float32).t())
    r = torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + EPS)
    return proj.view(s, b, -1), r.view(s, b, 1)


def _module(x, w):
    from skyrl.backends.skyrl_train.patches.megatron.mcore_ext.hyper_connection import (
        RMSNormInputHyperConnectionModule,
    )

    module = object.__new__(RMSNormInputHyperConnectionModule)  # only the projection's state
    object.__setattr__(module, "mapping_proj", types.SimpleNamespace(weight=w))
    object.__setattr__(module, "norm_eps", EPS)
    return module._projection_and_get_norm(x)


def _run(fn, x, w, grads):
    """Forward + backward; returns outputs, input grads and what autograd saved."""
    x, w = x.clone().requires_grad_(), w.clone().requires_grad_()
    saved = []

    def pack(t):
        saved.append((t.dtype, tuple(t.shape)))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        proj, r = fn(x, w)
    used = [(o, g) for o, g in zip((proj, r), grads) if g is not None]
    torch.autograd.backward([o for o, _ in used], [g for _, g in used])
    return (proj.detach(), r.detach()), (x.grad, w.grad), saved


@pytest.mark.parametrize("scale", [1e-3, 1.0, 30.0])
@pytest.mark.parametrize("used", ["both", "proj", "r"])
def test_matches_plain_autograd_bitwise(scale, used):
    gen = torch.Generator().manual_seed(0)
    x = (torch.randn(S, B, WIDTH, generator=gen) * scale).bfloat16()
    w = torch.randn(OUT, WIDTH, generator=gen) * 0.02
    g_proj, g_r = torch.randn(S, B, OUT, generator=gen), torch.randn(S, B, 1, generator=gen)
    grads = (g_proj if used != "r" else None, g_r if used != "proj" else None)

    (p0, r0), (gx0, gw0), saved0 = _run(_plain, x, w, grads)
    (p1, r1), (gx1, gw1), saved1 = _run(_module, x, w, grads)

    assert torch.equal(p0, p1) and torch.equal(r0, r1)
    assert torch.equal(gx0, gx1) and gx1.dtype == torch.bfloat16
    assert (gw0 is None) == (gw1 is None)
    if gw0 is not None:
        assert torch.equal(gw0, gw1)

    fp32_input_copy = (torch.float32, (S * B, WIDTH))
    assert fp32_input_copy in saved0  # what the checkpoint exists to avoid
    assert fp32_input_copy not in saved1
    assert (torch.bfloat16, (S * B, WIDTH)) in saved1
