"""Dequantization of the Inkling2 routed-expert weights.

The released checkpoint stores ``model.llm.layers.N.mlp.experts.{w13,w2}_weight`` in one of two
packed formats (selected per tensor by ``hf_quant_config.json``); every other tensor is plain
BF16 / FP32. Logical shapes are ``w13 [E, 2 * I, H]`` (gate / up rows interleaved: even rows are
the gate, odd rows the up projection) and ``w2 [E, H, I]``.

NVFP4 (``uint8`` weight): two E2M1 codes per byte along the input dim, low nibble first; bit 3 of
a code is the sign and bits 0-2 index ``E2M1_VALUES``. Each 1x16 block has an E4M3 scale
(``*_weight_scale [E, N, K / 16]``) and each expert a global factor
``*_weight_amax[e] / *_weight_outer_scale_denominator``. Reference: the SGLang fork's
``sglang/tml/kernels/nvfp4.py`` (``_scale_output``: ``acc * x_amax * w_amax / (1536 * denom)``,
where activations carry ``x_amax / 1536``) and the packing in
``sglang/tml/tests/test_inkling2_nvfp4.py``.

MXFP8 (``float8_e4m3fn`` weight): each 1x32 block along the input dim has a UE8M0 exponent
(``*_weight_scale [E, N, K / 32]``, ``uint8``): ``W = q * 2^(s - 127)``.

The MXFP8 dequantization is exact in BF16. NVFP4 codes times their E4M3 block scale are exact
too; the per-expert global factor (an arbitrary FP32) adds one BF16 rounding per weight, which
the serving kernel instead applies in FP32 in the GEMM epilogue.
"""

from typing import Dict

import torch

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
SCALE_SUFFIXES = ("_scale", "_amax", "_outer_scale_denominator")


def _e2m1_table(device: torch.device) -> torch.Tensor:
    """FP32 value of every 4-bit code (sign in bit 3)."""
    magnitude = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
    return torch.cat([magnitude, -magnitude])


def unpack_nvfp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """``uint8 [..., K / 2]`` -> code indices ``uint8 [..., K]`` (low nibble = even column)."""
    return torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)


def dequantize_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor,
    outer_scale_denominator: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize ``packed [E, N, K / 2]`` with ``scale [E, N, K / 16]`` (E4M3), ``amax [E]``."""
    values = _e2m1_table(packed.device)[unpack_nvfp4_codes(packed).long()]
    values = values.unflatten(-1, (-1, NVFP4_BLOCK)) * scale.float().unsqueeze(-1)
    factor = amax.float() / outer_scale_denominator.float()
    return (values.flatten(-2) * factor.view(-1, *([1] * (packed.dim() - 1)))).to(dtype)


def dequantize_mxfp8(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize ``weight [E, N, K]`` (E4M3) with UE8M0 ``scale [E, N, K / 32]``."""
    exponent = torch.exp2(scale.float() - 127.0).unsqueeze(-1)
    return (weight.float().unflatten(-1, (-1, MXFP8_BLOCK)) * exponent).flatten(-2).to(dtype)


def dequantize_expert_weight(
    name: str, tensors: Dict[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Dequantize the expert weight ``name`` (e.g. ``...experts.w13_weight``) from ``tensors``,
    which must also hold its ``name + suffix`` scale tensors. A BF16 weight is returned as is."""
    weight = tensors[name]
    if weight.dtype == torch.uint8:
        return dequantize_nvfp4(
            weight,
            tensors[name + "_scale"],
            tensors[name + "_amax"],
            tensors[name + "_outer_scale_denominator"],
            dtype,
        )
    if weight.dtype == torch.float8_e4m3fn:
        return dequantize_mxfp8(weight, tensors[name + "_scale"], dtype)
    return weight.to(dtype)


def nvfp4_requantization_mismatches(
    dequantized: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor,
    outer_scale_denominator: torch.Tensor,
) -> int:
    """Count codes that do not survive re-quantizing ``dequantized`` with the stored scales.

    Dividing by the block scale and global factor and snapping to the nearest E2M1 value must
    give back every stored code: the BF16 rounding of the dequantized weight is far below the
    E2M1 grid spacing. A nonzero count means the nibble order, block layout or scale formula is
    wrong. Zero-scale blocks (all codes decode to 0) are skipped since their codes are arbitrary.
    """
    factor = (amax.float() / outer_scale_denominator.float()).view(-1, *([1] * (packed.dim() - 1)))
    block_scale = scale.float().repeat_interleave(NVFP4_BLOCK, dim=-1)
    divisor = block_scale * factor
    ratio = dequantized.float() / torch.where(divisor == 0, 1.0, divisor)
    grid = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    nearest = grid[torch.bucketize(ratio.abs(), (grid[:-1] + grid[1:]) / 2)].copysign(ratio)
    stored = _e2m1_table(packed.device)[unpack_nvfp4_codes(packed).long()]
    # Compared as values: +0 and -0 (codes 0 and 8) decode to the same weight.
    same = (nearest == stored) | (divisor == 0)
    return int((~same).sum())


def split_gate_up(w13: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """Split interleaved ``w13 [..., 2 * I, H]`` into ``(gate [..., I, H], up [..., I, H])``."""
    return w13[..., 0::2, :], w13[..., 1::2, :]


def is_expert_scale_tensor(name: str) -> bool:
    return ".mlp.experts." in name and name.endswith(SCALE_SUFFIXES)
