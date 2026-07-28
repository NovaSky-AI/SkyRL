"""CPU unit tests for the fake-INT4 QAT straight-through estimator.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest -s tests/backends/skyrl_train/test_fake_int4_qat.py

The golden tests pin the STE to slices of REAL quantization artifacts (see
``_fake_int4_qat_golden.py``): any change to the grid arithmetic (fp32 vs bf16
rounding, eps clamps, divisor/clamp-range handling) that diverges from what the
inference engine serves fails these tests bit-exactly, without needing
compressed-tensors, megatron, or a GPU.
"""

import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron.fake_int4_qat import (
    _FakeInt4QuantizeSTE,
    fake_int4_quantize_ste,
)
from tests.backends.skyrl_train._fake_int4_qat_golden import KIMI_QAT, QWEN_RTN

GS = 32

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _unpack_int4(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """compressed-tensors unpack_from_int32 semantics (offset encoding: nibble = q + 8)."""
    out = torch.zeros(rows, packed.shape[1] * 8, dtype=torch.int32)
    for j in range(8):
        out[:, j::8] = (packed >> (4 * j)) & 0xF
    return (out - 8).to(torch.int8)[:, :cols]


def _load_golden(case: dict, device: str):
    rows, cols = case["rows"], case["cols"]
    masters = torch.tensor(case["masters_u16"], dtype=torch.uint16).view(torch.bfloat16).reshape(rows, cols)
    packed = torch.tensor(case["packed_i32"], dtype=torch.int32)
    scale = torch.tensor(case["scale_u16"], dtype=torch.uint16).view(torch.bfloat16).reshape(rows, cols // GS)
    q_ref = _unpack_int4(packed, rows, cols)
    # the served grid, exactly as compressed-tensors dequantize() produces it
    served = (q_ref.view(rows, cols // GS, GS).to(torch.bfloat16) * scale.unsqueeze(-1)).reshape(rows, cols)
    return masters.to(device), q_ref.to(device), scale.to(device), served.to(device)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("case", [QWEN_RTN, KIMI_QAT], ids=["qwen_rtn_7p5", "kimi_qat_7p0"])
def test_golden_bit_exact_vs_served_artifact(case, device):
    """STE output must equal the served INT4 grid bit-for-bit on real checkpoint slices."""
    masters, q_ref, scale_ref, served = _load_golden(case, device)
    deq = fake_int4_quantize_ste(masters, case["group_size"], case["scale_divisor"], case["q_min"])
    assert deq.dtype == torch.bfloat16
    assert torch.equal(deq, served), "dequantized weights diverge from the served artifact grid"

    # the recomputed scale and codes must match the stored artifact too
    rows, cols = case["rows"], case["cols"]
    g = masters.view(rows, cols // GS, GS)
    amax = g.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (amax / case["scale_divisor"]).to(masters.dtype)
    assert torch.equal(scale.squeeze(-1), scale_ref), "recomputed scale != stored weight_scale"
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(g / safe), case["q_min"], 7.0)
    assert torch.equal(q.to(torch.int8).view(rows, cols), q_ref), "recomputed codes != stored codes"


def test_golden_covers_edge_cases():
    """The fixtures must keep covering the regressions they were built to catch."""
    _, q_qwen, _, _ = _load_golden(QWEN_RTN, "cpu")
    assert (q_qwen == -8).any(), "RTN slice lost its stored -8 code coverage"
    masters, _, _, _ = _load_golden(QWEN_RTN, "cpu")
    amax = masters.view(QWEN_RTN["rows"], -1, GS).float().abs().amax(-1)
    assert (amax < 1e-30).any(), "RTN slice lost its near-denormal group coverage (eps-clamp regression guard)"
    _, q_kimi, _, _ = _load_golden(KIMI_QAT, "cpu")
    mx = q_kimi.view(KIMI_QAT["rows"], -1, GS).abs().amax(-1)
    assert mx.eq(7).all(), "Kimi slice must have max|q| == 7 in every group"


@pytest.mark.parametrize("device", _DEVICES)
def test_kimi_identity_on_dequant_masters(device):
    """div=7 property that makes the Kimi flow exact: fake-quant of a dequant is the identity."""
    masters, _, _, served = _load_golden(KIMI_QAT, device)
    assert torch.equal(masters, served)  # masters ARE the dequant for this fixture
    deq = fake_int4_quantize_ste(masters, GS, 7.0, -7.0)
    assert torch.equal(deq, masters)
    # and it is idempotent in general for freshly quantized tensors
    w = torch.randn(16, 128, dtype=torch.bfloat16, device=device)
    once = fake_int4_quantize_ste(w, GS, 7.0, -7.0)
    twice = fake_int4_quantize_ste(once, GS, 7.0, -7.0)
    assert torch.equal(once, twice)


@pytest.mark.parametrize("device", _DEVICES)
def test_ste_backward_is_identity(device):
    w = torch.randn(8, 64, dtype=torch.bfloat16, device=device, requires_grad=True)
    out = fake_int4_quantize_ste(w, GS, 7.5, -8.0)
    grad = torch.randn_like(out)
    out.backward(grad)
    assert torch.equal(w.grad, grad)


def test_main_grad_attribute_propagates():
    w = torch.randn(8, 64, dtype=torch.bfloat16)
    w.main_grad = torch.zeros(8, 64)
    out = fake_int4_quantize_ste(w, GS, 7.5, -8.0)
    assert out.main_grad is w.main_grad
    # and absent when the master has none
    w2 = torch.randn(8, 64, dtype=torch.bfloat16)
    assert not hasattr(fake_int4_quantize_ste(w2, GS, 7.5, -8.0), "main_grad")


@pytest.mark.parametrize("device", _DEVICES)
def test_padding_path_matches_manual_padding(device):
    """in_features % group_size != 0: pad with zeros, quantize, slice back."""
    w = torch.randn(8, 40, dtype=torch.bfloat16, device=device)
    out = fake_int4_quantize_ste(w, GS, 7.5, -8.0)
    assert out.shape == w.shape and torch.isfinite(out).all()
    padded = torch.zeros(8, 64, dtype=torch.bfloat16, device=device)
    padded[:, :40] = w
    ref = fake_int4_quantize_ste(padded, GS, 7.5, -8.0)[:, :40]
    assert torch.equal(out, ref)


@pytest.mark.parametrize("device", _DEVICES)
def test_all_zero_group_quantizes_to_zero(device):
    w = torch.zeros(4, 64, dtype=torch.bfloat16, device=device)
    w[:, GS:] = torch.randn(4, GS, dtype=torch.bfloat16, device=device)
    out = fake_int4_quantize_ste(w, GS, 7.5, -8.0)
    assert torch.isfinite(out).all(), "all-zero group must not produce NaN (0/0 guard)"
    assert out[:, :GS].abs().sum() == 0


@pytest.mark.parametrize(("div", "qmin"), [(7.5, -8.0), (7.0, -7.0)])
def test_codes_stay_in_convention_range(div, qmin):
    w = torch.randn(32, 256, dtype=torch.bfloat16)
    g = w.view(32, -1, GS)
    scale = (g.abs().amax(-1, keepdim=True).to(torch.float32) / div).to(w.dtype)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.round(g / safe).clamp(qmin, 7.0)
    deq = fake_int4_quantize_ste(w, GS, div, qmin)
    assert torch.equal(deq, (codes * scale).reshape(32, 256))
    assert codes.min() >= qmin and codes.max() <= 7.0
    if div == 7.0:
        # /7 pins the max-|w| element of each group to +-7 and never emits -8
        assert codes.abs().amax(-1).eq(7).all()


def test_non_contiguous_input_matches_contiguous():
    base = torch.randn(16, 256, dtype=torch.bfloat16)
    nc = base[:, ::2]
    assert not nc.is_contiguous()
    out = fake_int4_quantize_ste(nc, GS, 7.5, -8.0)
    ref = fake_int4_quantize_ste(nc.contiguous(), GS, 7.5, -8.0)
    assert torch.equal(out, ref)


def test_autograd_function_arity():
    """backward must return one gradient per forward input (x, group_size, scale_div, q_min)."""
    w = torch.randn(4, 32, dtype=torch.bfloat16, requires_grad=True)
    out = _FakeInt4QuantizeSTE.apply(w, GS, 7.5, -8.0)
    out.sum().backward()
    assert w.grad is not None
