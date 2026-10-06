"""``proj_rms_saving_input_dtype`` matches plain autograd bit for bit, without saving the FP32 upcast.

The mHC projection + RMS factor run in FP32 on a bf16 residual stream. Plain autograd keeps the
FP32 upcast of the input for backward (2 GiB per mHC site at 32k tokens per rank on
GLM-5.3-Flash); ``mcore_ext/hyper_connection.py`` saves only the bf16 input and reruns the FP32
math in backward. Both input-norm placements (``eps_inside_sqrt``) are checked, at several
activation scales, with both outputs or only one of them feeding the loss.

Pure tensor math, so it runs on CPU and lives outside ``gpu/``; the module-level comparison
against HF is ``gpu_ci/patches/megatron/mcore_ext/test_modules_vs_hf.py``.
"""

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")

# Runs in the CPU megatron job (`-m megatron`); without the marker that job deselects it.
pytestmark = pytest.mark.megatron

TOKENS, WIDTH, OUT = 512, 4 * 256, 24  # n=4 streams; OUT = n^2 + 2n mapping columns


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


@pytest.mark.parametrize("eps_inside_sqrt", [True, False])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 30.0])
@pytest.mark.parametrize("used", ["both", "proj", "r"])
def test_matches_plain_autograd_bitwise(eps_inside_sqrt, scale, used):
    from skyrl.backends.skyrl_train.patches.megatron.mcore_ext.hyper_connection import (
        _proj_rms_fp32,
        proj_rms_saving_input_dtype,
    )

    gen = torch.Generator().manual_seed(0)
    x = (torch.randn(TOKENS, WIDTH, generator=gen) * scale).bfloat16()
    w = torch.randn(OUT, WIDTH, generator=gen) * 0.02
    g_proj, g_r = torch.randn(TOKENS, OUT, generator=gen), torch.randn(TOKENS, 1, generator=gen)
    grads = (g_proj if used != "r" else None, g_r if used != "proj" else None)
    eps = 1e-5

    def plain(x, w):
        return _proj_rms_fp32(x.to(torch.float32), w.to(torch.float32), eps, eps_inside_sqrt)

    def patched(x, w):
        return proj_rms_saving_input_dtype(x, w, eps, eps_inside_sqrt)

    (p0, r0), (gx0, gw0), saved0 = _run(plain, x, w, grads)
    (p1, r1), (gx1, gw1), saved1 = _run(patched, x, w, grads)

    assert torch.equal(p0, p1) and torch.equal(r0, r1)
    assert torch.equal(gx0, gx1) and gx1.dtype == torch.bfloat16
    assert (gw0 is None) == (gw1 is None)
    if gw0 is not None:
        assert torch.equal(gw0, gw1)

    fp32_input_copy = (torch.float32, (TOKENS, WIDTH))
    assert fp32_input_copy in saved0  # what the patch exists to avoid
    assert fp32_input_copy not in saved1
    assert (torch.bfloat16, (TOKENS, WIDTH)) in saved1
