"""The MXFP8 wire quantizer must stay bitwise identical to Transformer Engine.

On Blackwell the trainer holds BF16 primaries and TE re-quantizes per GEMM, so
rollout and trainer agree only because both derive the same codes from the same
tensor. That identity is an invariant of this sync path, not an observation:
if TE's rule ever moves, the rollout silently samples from a different policy
than the one being optimized. Hence a standing test rather than a one-off check.

Run: uv run --isolated --extra dev --extra megatron pytest \
       tests/backends/skyrl_train/gpu/gpu_ci/test_mxfp8_te_parity.py
"""

import importlib
import os

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    is_blackwell_or_newer,
)
from skyrl.backends.skyrl_train.weight_sync.fp8 import (
    MXFP8_GROUP_SIZE,
    batched_mx_cast_to_fp8,
    mx_cast_to_fp8,
)

pytestmark = pytest.mark.skipif(
    not is_blackwell_or_newer(),
    reason="TE's MXFP8 quantizer requires a Blackwell (SM100+) device",
)


def _te_reference(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # transformer_engine_torch is not a top-level module: it ships inside
    # transformer_engine/wheel_lib, which transformer_engine's __init__ appends to
    # sys.path. Importing it first therefore fails, so import the package before
    # it and keep them in separate statements that an import sorter cannot reorder
    # into the broken sequence.
    import transformer_engine.pytorch  # noqa: F401  (registers wheel_lib on sys.path)
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    tex = importlib.import_module("transformer_engine_torch")

    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    quantized = quantizer(weight.clone())
    return quantized._rowwise_data, quantized._rowwise_scale_inv


def _te_scales_unpadded(te_scales: torch.Tensor, rows: int, num_groups: int) -> torch.Tensor:
    """Drop TE's swizzle padding: rows round up to 128, groups to 4."""

    flat = te_scales.flatten()
    padded_groups = ((num_groups + 3) // 4) * 4
    padded_rows = flat.numel() // padded_groups
    return flat.view(padded_rows, padded_groups)[:rows, :num_groups]


def _assert_matches_te(weight: torch.Tensor) -> None:
    codes, scales = mx_cast_to_fp8(weight)
    te_codes, te_scales = _te_reference(weight)
    rows, num_groups = scales.shape
    assert torch.equal(codes.view(torch.uint8), te_codes.view(torch.uint8).view(codes.shape))
    assert torch.equal(scales, _te_scales_unpadded(te_scales, rows, num_groups))


@pytest.mark.parametrize(
    "shape,dtype,scale",
    [
        ((256, 512), torch.bfloat16, 1.0),
        ((128, 256), torch.float32, 1.0),
        ((64, 128), torch.bfloat16, 1e-6),
        ((64, 128), torch.bfloat16, 1e4),
        ((32, 64), torch.bfloat16, 1.0),
    ],
)
def test_mx_cast_matches_te_bitwise(shape, dtype, scale):
    torch.manual_seed(0)
    weight = torch.randn(*shape, device="cuda", dtype=dtype) * scale
    _assert_matches_te(weight)


def test_mx_cast_matches_te_for_zero_token_expert_blocks():
    """MoE experts that receive no tokens produce all-zero groups."""

    torch.manual_seed(0)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    weight[:, :MXFP8_GROUP_SIZE] = 0.0
    _assert_matches_te(weight)
    _assert_matches_te(torch.zeros(32, 64, device="cuda", dtype=torch.bfloat16))


def test_batched_mx_cast_matches_te_per_expert():
    torch.manual_seed(0)
    experts = torch.randn(4, 128, 256, device="cuda", dtype=torch.bfloat16)
    codes, scales = batched_mx_cast_to_fp8(experts, expert_batch_size=2)
    for i in range(experts.shape[0]):
        te_codes, te_scales = _te_reference(experts[i])
        rows, num_groups = scales[i].shape
        assert torch.equal(codes[i].view(torch.uint8), te_codes.view(torch.uint8).view(codes[i].shape))
        assert torch.equal(scales[i], _te_scales_unpadded(te_scales, rows, num_groups))


# --- TE fast-path dispatch (quantize.py routes CUDA casts through TE) ---


def _forced_python(fn, *args, **kwargs):
    """Run a cast with the TE fast path disabled, restoring dispatch after."""

    import skyrl.backends.skyrl_train.weight_sync.fp8.quantize as quantize_mod

    monkey = os.environ.get("SKYRL_MX_CAST_BACKEND")
    os.environ["SKYRL_MX_CAST_BACKEND"] = "python"
    quantize_mod._TE_MX_PROBED = False
    try:
        return fn(*args, **kwargs)
    finally:
        if monkey is None:
            os.environ.pop("SKYRL_MX_CAST_BACKEND", None)
        else:
            os.environ["SKYRL_MX_CAST_BACKEND"] = monkey
        quantize_mod._TE_MX_PROBED = False


def test_te_fast_path_is_active_and_bitwise_equal_2d():
    import skyrl.backends.skyrl_train.weight_sync.fp8.quantize as quantize_mod

    assert quantize_mod._te_mx_quantizer() is not None, "TE fast path should engage on SM100+"
    torch.manual_seed(0)
    weight = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
    codes_te, scales_te = mx_cast_to_fp8(weight)
    codes_py, scales_py = _forced_python(mx_cast_to_fp8, weight)
    assert torch.equal(codes_te.view(torch.uint8), codes_py.view(torch.uint8))
    assert torch.equal(scales_te, scales_py)


def test_te_fast_path_flattened_experts_bitwise_equal_3d():
    torch.manual_seed(0)
    experts = torch.randn(8, 256, 512, device="cuda", dtype=torch.bfloat16)
    experts[3].zero_()  # zero-token expert
    codes_te, scales_te = batched_mx_cast_to_fp8(experts)
    codes_py, scales_py = _forced_python(batched_mx_cast_to_fp8, experts)
    assert torch.equal(codes_te.view(torch.uint8), codes_py.view(torch.uint8))
    assert torch.equal(scales_te, scales_py)


def test_te_fast_path_row_guard_falls_back():
    """TE's kernel asserts rows % 32 == 0; such tensors must take the torch path."""

    torch.manual_seed(0)
    weight = torch.randn(137, 320, device="cuda", dtype=torch.bfloat16)
    codes, scales = mx_cast_to_fp8(weight)  # must not raise
    codes_py, scales_py = _forced_python(mx_cast_to_fp8, weight)
    assert torch.equal(codes.view(torch.uint8), codes_py.view(torch.uint8))
    assert torch.equal(scales, scales_py)


def test_te_fast_path_subnormal_flush_is_pinned():
    """All-subnormal groups: TE flushes codes to zero, the torch oracle quantizes.

    Out of contract (trained BF16 weights never produce a group whose every
    element is below 2**-126) but pinned so a behavior change is noticed.
    """

    weight = torch.full((128, 64), 1e-38, device="cuda", dtype=torch.bfloat16)
    codes_te, scales_te = mx_cast_to_fp8(weight)
    codes_py, scales_py = _forced_python(mx_cast_to_fp8, weight)
    assert torch.equal(scales_te, scales_py)
    assert (codes_te.view(torch.uint8) == 0).all()
    assert not (codes_py.view(torch.uint8) == 0).all()
