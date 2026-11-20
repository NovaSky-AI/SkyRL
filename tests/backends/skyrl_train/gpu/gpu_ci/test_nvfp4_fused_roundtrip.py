"""GPU tests for the fused NVFP4 round-trip kernels (``kernels/nvfp4_quant/fused_rt.py``, sm_100).

The fused kernels must be bit-identical to FlashInfer's quantizers (``fp4_quantize`` for expert weights,
``nvfp4_quantize(per_token_activation=True)`` for activations, absmax and the 4/6 MSE fast-error contract) and to
the decode of their bytes: codes, linear and swizzled E4M3 scales, decode scales and the bf16 round trip. The
public APIs give the same bytes with the fused path on and off (``SKYRL_QAT_NVFP4_FUSED=0``, the FlashInfer fallback),
and the speculative expert amax stays exact when the weights change.

uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_nvfp4_fused_roundtrip.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="needs sm_100 (B200/B300)"
)
pytest.importorskip("flashinfer")

from skyrl.backends.skyrl_train.kernels.nvfp4_quant import (  # noqa: E402
    fake_experts,
    fused_rt,
    per_token,
)
from skyrl.backends.skyrl_train.kernels.nvfp4_quant.decode import (  # noqa: E402
    dequantize_rows,
)

BLOCK_SCALES = ("absmax", "four_over_six")
WEIGHT_KINDS = ("normal", "heavy", "tiny", "ties", "zeros", "subnormal", "nonfinite")


@pytest.fixture(autouse=True)
def _fused_on(monkeypatch):
    monkeypatch.delenv("SKYRL_QAT_NVFP4_FUSED", raising=False)
    if not fused_rt.available():
        pytest.fail("the fused NVFP4 kernels did not build on an sm_100 device")
    fused_rt.AMAX_CACHE.clear()


def make_weights(e, n, k, kind, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    o = dict(generator=g, device="cuda")
    w = torch.randn(e, n, k, **o) * 0.02
    if kind == "heavy":
        w = w * torch.exp(torch.randn(e, n, 1, **o))
        w[torch.rand(e, n, k, **o) < 2e-3] *= 30
    elif kind == "tiny":
        w = w * 1e-4
    elif kind == "ties":  # exact halves hit E2M1 midpoints and grid points
        w = (torch.randint(-12, 13, (e, n, k), **o).float() / 2) * 0.01
    elif kind == "zeros":  # all-zero, -0 and single-nonzero blocks, one all-zero expert
        w[:, ::3, :16] = 0.0
        w[:, 1::3, :16] = -0.0
        w[:, 2::3, 16:32] = 0.0
        w[:, 2::3, 20] = 0.01
        w[0] = 0.0
    elif kind == "subnormal":  # rows over 1e-6..1e-1 under one amax: E4M3-subnormal and zero block scales
        w = torch.randn(e, n, k, **o) * 10 ** (-6 + 5 * torch.rand(e, n, 1, **o))
        w[:, 0, 0] = 1.0
    elif kind == "nonfinite":  # an Inf and a NaN expert (gs = 0 / NaN): FlashInfer's bytes and their decode
        w[0, 3, 5] = float("inf")
        if e > 1:
            w[1, 7, 9] = float("nan")
    return w.to(torch.bfloat16)


def flashinfer_weights(w3, block_scale):
    """FlashInfer's bytes (linear and swizzled scales), alpha, and their decode."""
    e, n, k = w3.shape
    gs, alpha = fake_experts.global_scales(w3)
    q, sf = fake_experts.quantize(w3, gs, block_scale)
    _, swz = fake_experts.quantize(w3, gs, block_scale, swizzled=True)
    y = dequantize_rows(q.view(e * n, k // 2), sf.view(e * n, k // 16), alpha.repeat_interleave(n)).view(e, n, k)
    return y, q, sf, swz, alpha


def bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def assert_same(a, b, what):
    """Bit for bit; bf16 NaNs count as equal whatever their payload."""
    diff = bits(a) != bits(b)
    if a.dtype == torch.bfloat16:
        diff &= ~(torch.isnan(a) & torch.isnan(b))
    bad = int(diff.sum())
    assert bad == 0, f"{what}: {bad} of {a.numel()} differ"


@pytest.mark.parametrize("kind", WEIGHT_KINDS)
@pytest.mark.parametrize("shape", [(3, 128, 64), (2, 256, 192), (16, 512, 2048)])
@pytest.mark.parametrize("block_scale", BLOCK_SCALES)
def test_expert_kernel_is_flashinfer(kind, shape, block_scale):
    w3 = make_weights(*shape, kind)
    e, n, k = shape
    y_ref, q_ref, sf_ref, swz_ref, alpha_ref = flashinfer_weights(w3, block_scale)
    y = torch.empty_like(w3)
    q = torch.empty_like(q_ref)
    lin = torch.empty_like(sf_ref)
    swz = torch.empty_like(swz_ref)
    _, alpha = fused_rt.expert_quantize(
        w3, block_scale == "four_over_six", out=y, codes=q, sf_linear=lin, sf_swizzled=swz
    )
    assert_same(y, y_ref, "round trip")
    assert_same(q, q_ref, "codes")
    assert_same(lin, sf_ref, "linear scales")
    assert_same(swz, swz_ref.view(swz.shape), "swizzled scales")
    torch.testing.assert_close(alpha, alpha_ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("block_scale", BLOCK_SCALES)
def test_speculative_amax_is_exact(block_scale):
    w3 = make_weights(4, 256, 512, "heavy")
    fs = block_scale == "four_over_six"
    for step in range(4):
        if step == 2:
            w3[1].mul_(1.5)  # one expert changes (an optimizer step)
        if step == 3:
            w3.mul_(0.75)  # all change
        y_ref, *_ = flashinfer_weights(w3, block_scale)
        y = torch.empty_like(w3)
        fused_rt.expert_quantize(w3, fs, out=y, speculate=True)
        assert_same(y, y_ref, f"speculative round trip, step {step}")


def test_api_fused_and_fallback_agree(monkeypatch):
    weights = list(make_weights(4, 256, 512, "heavy").unbind(0))
    buffer = torch.empty(4 * weights[0].numel(), dtype=torch.bfloat16, device="cuda")  # a reversed DDP-like view
    in_buffer = [buffer[i * weights[0].numel() : (i + 1) * weights[0].numel()].view(weights[0].shape) for i in range(4)]
    in_buffer = in_buffer[::-1]
    for w, b in zip(weights, in_buffer):
        b.copy_(w)
    x = per_token_activations(3000, 2048, 1)
    dst = torch.randperm(3072, device="cuda")[:3000].sort().values.to(torch.int32)

    def run():
        out = {}
        for bs in BLOCK_SCALES:
            fs = bs == "four_over_six"
            for layout, ws in (("apart", weights), ("buffer", in_buffer)):
                out[f"rt/{bs}/{layout}"] = torch.stack(fake_experts.fake_quantize_expert_weights(ws, bs))
                p = fake_experts.pack_expert_weights(ws, bs, keep_linear=True)
                out[f"pack/{bs}/{layout}"] = (p.q, p.sf, p.alpha, p.sf_linear, p.reversed)
            out[f"act/{bs}"] = per_token.fake_quantize_per_token(x, four_over_six=fs)
            xi = x.clone()
            out[f"act_inplace/{bs}"] = per_token.fake_quantize_per_token(xi, inplace=True, four_over_six=fs)
            xw = x.clone()
            out[f"pack_act/{bs}"] = per_token.pack_per_token(xw, dst, 3072, fs, write_back=True) + (xw,)
        return out

    fused = run()
    monkeypatch.setenv("SKYRL_QAT_NVFP4_FUSED", "0")
    assert not fused_rt.available()
    fallback = run()
    rows = dst.long()
    for key, a in fused.items():
        b = fallback[key]
        if key.startswith("pack_act/"):  # padded rows' codes and decode scales are unspecified
            a, b = (a[0][rows], a[1], a[2][rows], a[3]), (b[0][rows], b[1], b[2][rows], b[3])
        for i, (u, v) in enumerate(zip(a, b) if isinstance(a, tuple) else [(a, b)]):
            if isinstance(u, torch.Tensor):
                assert_same(u, v, f"{key}[{i}]")
            else:
                assert u == v, key


def per_token_activations(m, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), generator=g, device="cuda")
    chi = torch.randn((m, k, 2), generator=g, device="cuda").pow(2).sum(-1) / 2
    x = x / chi.sqrt() * torch.exp(torch.randn((m, 1), generator=g, device="cuda"))  # Student-t(2), row scales
    x[::97] = 0  # all-zero rows
    x[1::101, :7] = 300.0  # outlier channels
    x[2::103, :16] = -0.0
    x[3::107] *= 1e-30  # E4M3-subnormal / zero block scales
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("k", [64, 512, 2048, 6144, 7168])
@pytest.mark.parametrize("block_scale", BLOCK_SCALES)
def test_per_token_kernel_is_flashinfer(k, block_scale):
    fs = block_scale == "four_over_six"
    wide = per_token_activations(1500, k + 64, 2)
    for x in (wide[:, :k].contiguous(), wide[:, :k]):  # contiguous and strided rows
        q_ref, sf_ref, t_ref = per_token.flashinfer_per_token(x, fs)
        y_ref = dequantize_rows(q_ref, sf_ref, t_ref)
        y = torch.empty_like(x)
        q, lin, t = torch.empty_like(q_ref), torch.empty_like(sf_ref), torch.empty_like(t_ref)
        assert fused_rt.per_token_quantize(x, fs, out=y, codes=q, sf_linear=lin, t=t)
        assert_same(y, y_ref, "round trip")
        assert_same(q, q_ref, "codes")
        assert_same(lin, sf_ref, "scales")
        assert torch.equal(t, t_ref)


@pytest.mark.parametrize("block_scale", BLOCK_SCALES)
def test_per_token_nonfinite_is_flashinfer(block_scale):
    """FlashInfer's amaxes skip NaN (``__hmax2`` / ``fmaxf``): a block's is NaN only when all 16 values are, a
    row's never; the decode reads the NaN scale byte as 480."""
    fs = block_scale == "four_over_six"
    x = per_token_activations(256, 2048, 4)
    x[0, 3] = float("nan")  # one NaN in a block
    x[1, 16:32] = float("nan")  # an all-NaN block
    x[2] = float("nan")  # an all-NaN row
    x[3, 40] = float("inf")
    x[4, :8] = float("nan")  # the even or odd half of a block
    x[4, 16:32:2] = float("nan")
    x[5, 7] = -float("inf")
    x[5, 9] = float("nan")
    q_ref, sf_ref, t_ref = per_token.flashinfer_per_token(x, fs)
    y_ref = dequantize_rows(q_ref, sf_ref, t_ref)
    y, q, lin, t = torch.empty_like(x), torch.empty_like(q_ref), torch.empty_like(sf_ref), torch.empty_like(t_ref)
    assert fused_rt.per_token_quantize(x, fs, out=y, codes=q, sf_linear=lin, t=t)
    assert_same(q, q_ref, "codes")
    assert_same(lin, sf_ref, "scales")
    assert torch.equal(t, t_ref)
    assert_same(y, y_ref, "round trip")


def test_per_token_too_wide_falls_back():
    assert not fused_rt.per_token_fits(8192 + 16)
    x = per_token_activations(64, 8192 + 64, 3)
    assert not fused_rt.per_token_quantize(x, True, out=torch.empty_like(x))
    y = per_token.fake_quantize_per_token(x, four_over_six=True)  # the FlashInfer path
    q, sf, t = per_token.flashinfer_per_token(x, True)
    assert_same(y, dequantize_rows(q, sf, t), "fallback round trip")
