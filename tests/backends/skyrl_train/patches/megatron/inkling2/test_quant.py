"""Dequantization of the Inkling2 NVFP4 / MXFP8 expert weights.

The expected values are built element by element from the format description in
``patches/megatron/inkling2/quant.py`` (taken from the SGLang fork's kernels), independently of
the vectorized implementation, so a nibble-order, block-layout or scale mix-up shows up here
rather than as a logprob mismatch on the real checkpoint.
"""

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")

from skyrl.backends.skyrl_train.patches.megatron.inkling2.quant import (  # noqa: E402
    E2M1_VALUES,
    dequantize_expert_weight,
    dequantize_mxfp8,
    dequantize_nvfp4,
    nvfp4_requantization_mismatches,
    split_gate_up,
)

pytestmark = pytest.mark.megatron

E, N, K = 3, 4, 64


def _code_value(code: int) -> float:
    return (-1.0 if code & 8 else 1.0) * E2M1_VALUES[code & 7]


def _random_nvfp4(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (E, N, K), generator=g, dtype=torch.uint8)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    scale = (torch.rand(E, N, K // 16, generator=g) * 4 + 0.1).to(torch.float8_e4m3fn)
    amax = torch.rand(E, generator=g) * 10 + 0.5
    denominator = torch.tensor([2688.0])
    return codes, packed, scale, amax, denominator


def test_nvfp4_matches_elementwise_formula():
    codes, packed, scale, amax, denominator = _random_nvfp4()
    expected = torch.empty(E, N, K)
    for e in range(E):
        for n in range(N):
            for k in range(K):
                expected[e, n, k] = (
                    _code_value(int(codes[e, n, k])) * float(scale[e, n, k // 16].float()) * float(amax[e]) / 2688.0
                )
    got = dequantize_nvfp4(packed, scale, amax, denominator, dtype=torch.float32)
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=0)
    bf16 = dequantize_nvfp4(packed, scale, amax, denominator)
    assert bf16.dtype == torch.bfloat16
    # One BF16 rounding of the global factor: at most half a BF16 ulp.
    torch.testing.assert_close(bf16.float(), expected, rtol=2**-8, atol=0)


def test_nvfp4_requantization_roundtrip_detects_layout_errors():
    _, packed, scale, amax, denominator = _random_nvfp4(1)
    dequantized = dequantize_nvfp4(packed, scale, amax, denominator)
    assert nvfp4_requantization_mismatches(dequantized, packed, scale, amax, denominator) == 0
    swapped = (packed >> 4) | (packed << 4)
    assert nvfp4_requantization_mismatches(dequantized, swapped, scale, amax, denominator) > 0


def test_mxfp8_matches_elementwise_formula():
    g = torch.Generator().manual_seed(2)
    weight = (torch.randn(E, N, K, generator=g) * 50).to(torch.float8_e4m3fn)
    scale = torch.randint(110, 135, (E, N, K // 32), generator=g, dtype=torch.uint8)
    expected = weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)
    got = dequantize_mxfp8(weight, scale)
    # Exact: an E4M3 value times a power of two is representable in BF16.
    torch.testing.assert_close(got, expected.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(got.float(), expected, rtol=0, atol=0)


def test_dequantize_expert_weight_dispatch_and_gate_up_split():
    _, packed, scale, amax, denominator = _random_nvfp4(3)
    name = "model.llm.layers.0.mlp.experts.w13_weight"
    tensors = {
        name: packed,
        name + "_scale": scale,
        name + "_amax": amax,
        name + "_outer_scale_denominator": denominator,
    }
    w13 = dequantize_expert_weight(name, tensors)
    torch.testing.assert_close(w13, dequantize_nvfp4(packed, scale, amax, denominator), rtol=0, atol=0)
    gate, up = split_gate_up(w13)
    torch.testing.assert_close(gate[:, 1], w13[:, 2], rtol=0, atol=0)
    torch.testing.assert_close(up[:, 1], w13[:, 3], rtol=0, atol=0)
