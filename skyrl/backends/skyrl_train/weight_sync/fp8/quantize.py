"""FP8 quantization kernels and scale-mode helpers for blockwise and MXFP8."""

from __future__ import annotations

import os
from operator import index
from typing import Any, Sequence

import torch

# MXFP8 (OCP microscaling) groups 32 consecutive elements along the reduction
# dimension and stores one E8M0 exponent per group.
MXFP8_GROUP_SIZE = 32
E8M0_BIAS = 127
# Widest exponent an E8M0 byte can carry once biased; 255 is the NaN encoding.
_E8M0_MIN_EXP = -E8M0_BIAS
_E8M0_MAX_EXP = 254 - E8M0_BIAS


def use_power_2_scales_default() -> bool:
    """Return whether rollout weights use power-of-two block scales.

    The setting must match Transformer Engine. Hopper defaults to FP32 scales;
    Blackwell launchers select power-of-two scales by setting
    ``NVTE_FP8_BLOCK_SCALING_FP32_SCALES=0``.
    """

    scale_mode = os.getenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    if scale_mode not in {"0", "1"}:
        raise ValueError(
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES must be '0' (power-of-2) " f"or '1' (FP32 scales), got {scale_mode!r}"
        )
    return scale_mode == "0"


def normalize_block_size(block_size: Sequence[int]) -> tuple[int, int]:
    try:
        raw_values = tuple(block_size)
        if any(isinstance(value, bool) for value in raw_values):
            raise TypeError
        values = tuple(index(value) for value in raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weight_block_size must contain exactly two positive integers, got {block_size!r}") from exc
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError(f"weight_block_size must contain exactly two positive integers, got {block_size!r}")
    return values


# --------------------------------------------------------------------------
# Blockwise wire: 128x128 tiles, FP32 (or power-of-2) inverse scales.
# --------------------------------------------------------------------------


def blockwise_cast_to_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    power_2_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to vLLM's blockwise E4M3 checkpoint format.

    Returns ``weight_scale_inv`` such that
    ``weight ~= qweight.float() * scale``. Power-of-two mode rounds scales up
    to match Transformer Engine's UE8M0 rule.
    """

    if weight.ndim != 2:
        raise ValueError(f"Blockwise FP8 expects a 2D tensor, got shape={tuple(weight.shape)}")

    block_m, block_n = normalize_block_size(block_size)
    rows, cols = weight.shape
    padded_rows = ((rows + block_m - 1) // block_m) * block_m
    padded_cols = ((cols + block_n - 1) // block_n) * block_n

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    weight_fp32 = weight.detach().to(torch.float32).contiguous()
    if padded_rows != rows or padded_cols != cols:
        padded = weight_fp32.new_zeros((padded_rows, padded_cols))
        padded[:rows, :cols].copy_(weight_fp32)
    else:
        padded = weight_fp32

    blocks = padded.view(padded_rows // block_m, block_m, padded_cols // block_n, block_n)
    blocks = blocks.permute(0, 2, 1, 3)
    # Nonzero floor keeps all-zero blocks from degenerating the scale.
    scale = blocks.abs().amax(dim=(2, 3)).clamp(min=1e-10) / fp8_info.max
    if power_2_scale:
        # Rounding up preserves range and matches TE's power-of-two scale rule.
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    q_blocks = (blocks / scale[:, :, None, None]).clamp(min=fp8_info.min, max=fp8_info.max)
    q_blocks = q_blocks.to(torch.float8_e4m3fn)
    q_padded = q_blocks.permute(0, 2, 1, 3).contiguous().view(padded_rows, padded_cols)
    q_weight = q_padded[:rows, :cols].contiguous()
    return q_weight, scale.to(torch.float32).contiguous()


def batched_blockwise_cast_to_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    power_2_scale: bool = False,
    expert_batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D ``[experts, rows, cols]`` tensor blockwise.

    Quantizing several experts per operation avoids launching the full 2D
    conversion pipeline once per expert, while bounded batches limit peak FP32
    workspace.
    """

    if weight.ndim != 3:
        raise ValueError(f"Batched blockwise FP8 expects a 3D tensor, got shape={tuple(weight.shape)}")
    if isinstance(expert_batch_size, bool) or not isinstance(expert_batch_size, int) or expert_batch_size <= 0:
        raise ValueError(f"expert_batch_size must be a positive integer, got {expert_batch_size!r}")

    block_m, block_n = normalize_block_size(block_size)
    num_experts, rows, cols = weight.shape
    padded_rows = ((rows + block_m - 1) // block_m) * block_m
    padded_cols = ((cols + block_n - 1) // block_n) * block_n
    row_blocks = padded_rows // block_m
    col_blocks = padded_cols // block_n

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    q_weight = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty(
        (num_experts, row_blocks, col_blocks),
        dtype=torch.float32,
        device=weight.device,
    )

    for start in range(0, num_experts, expert_batch_size):
        end = min(start + expert_batch_size, num_experts)
        weight_fp32 = weight[start:end].detach().to(torch.float32).contiguous()
        if padded_rows != rows or padded_cols != cols:
            padded = weight_fp32.new_zeros((end - start, padded_rows, padded_cols))
            padded[:, :rows, :cols].copy_(weight_fp32)
        else:
            padded = weight_fp32

        blocks = padded.view(end - start, row_blocks, block_m, col_blocks, block_n)
        blocks = blocks.permute(0, 1, 3, 2, 4)
        scale = blocks.abs().amax(dim=(3, 4)).clamp(min=1e-10) / fp8_info.max
        if power_2_scale:
            scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        q_blocks = (blocks / scale[:, :, :, None, None]).clamp(min=fp8_info.min, max=fp8_info.max)
        q_blocks = q_blocks.to(torch.float8_e4m3fn)
        q_padded = q_blocks.permute(0, 1, 3, 2, 4).contiguous().view(end - start, padded_rows, padded_cols)
        q_weight[start:end].copy_(q_padded[:, :rows, :cols])
        scales[start:end].copy_(scale)

    return q_weight, scales


# --------------------------------------------------------------------------
# MXFP8 wire: 1x32 groups along the reduction dim, E8M0 exponent scales.
# Torch implementation is the parity oracle; TE's kernel is the fast path.
# --------------------------------------------------------------------------


def _mx_shared_exponent(amax: torch.Tensor) -> torch.Tensor:
    """Return the E8M0 shared exponent for each group.

    Transformer Engine derives it as ``ceil(log2(amax / 448))`` -- the same
    power-of-two rule the blockwise path applies to its FP32 scales. Groups that
    are entirely zero have no representable exponent; they take the minimum so
    every code in the group quantizes to zero.
    """

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    exponent = torch.ceil(torch.log2(amax / fp8_max))
    exponent = torch.where(amax > 0, exponent, torch.full_like(exponent, float(_E8M0_MIN_EXP)))
    return exponent.clamp(min=_E8M0_MIN_EXP, max=_E8M0_MAX_EXP)


_TE_MX_QUANTIZER: Any = None
_TE_MX_PROBED = False


def _te_mx_quantizer() -> Any:
    """Transformer Engine's MXFP8 quantizer, or ``None`` when unusable.

    TE's C++ kernel quantizes at memory bandwidth (~4 TB/s on B200) where the
    torch-composed implementation below runs at ~100 GB/s, so it is the default
    whenever it can run: CUDA tensor, TE importable, and the kernel actually
    working on this device (it requires SM100+). Bitwise equality between the
    two is enforced by tests/.../gpu_ci/test_mxfp8_te_parity.py — the torch
    implementation is the oracle, TE is the fast path.

    ``SKYRL_MX_CAST_BACKEND=python`` forces the torch path (escape hatch);
    ``=te`` makes an unusable TE an error instead of a silent fallback.
    """

    global _TE_MX_QUANTIZER, _TE_MX_PROBED
    backend = os.environ.get("SKYRL_MX_CAST_BACKEND", "auto").lower()
    if backend == "python":
        return None
    if _TE_MX_PROBED:
        if backend == "te" and _TE_MX_QUANTIZER is None:
            raise RuntimeError("SKYRL_MX_CAST_BACKEND=te but the TE MXFP8 quantizer is unusable on this device")
        return _TE_MX_QUANTIZER
    _TE_MX_PROBED = True
    try:
        import importlib

        import transformer_engine.pytorch  # noqa: F401  (registers wheel_lib on sys.path)
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        tex = importlib.import_module("transformer_engine_torch")
        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
        probe = torch.zeros(128, MXFP8_GROUP_SIZE, dtype=torch.bfloat16, device="cuda")
        quantizer(probe)
        _TE_MX_QUANTIZER = quantizer
    except Exception:
        _TE_MX_QUANTIZER = None
        if backend == "te":
            raise
    return _TE_MX_QUANTIZER


def _te_mx_cast_2d(weight: torch.Tensor, quantizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TE's quantizer and strip its padding back to the wire layout.

    TE pads the scale matrix (rows to a multiple of 128, groups to a multiple
    of 4); the wire ships the logical ``[rows, cols // 32]``.
    """

    rows, cols = weight.shape
    quantized = quantizer(weight)
    codes = quantized._rowwise_data
    if codes.dtype != torch.float8_e4m3fn:
        codes = codes.view(torch.float8_e4m3fn)
    scales = quantized._rowwise_scale_inv
    groups = cols // MXFP8_GROUP_SIZE
    if tuple(scales.shape) != (rows, groups):
        scales = scales[:rows, :groups].contiguous()
    return codes.view(rows, cols), scales


def mx_cast_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to MXFP8: E4M3 codes plus one E8M0 scale per 32 columns.

    Returns ``(codes, scales)`` where ``scales`` holds biased exponents as
    ``uint8`` in vLLM's compressed-tensors layout ``[rows, cols // 32]``, and
    ``weight ~= codes.float() * 2 ** (scales.int() - 127)``.

    The receiver requires the reduction dimension to be a multiple of the group
    size, so an unaligned tensor is a spec error rather than something to pad.
    """

    if weight.ndim != 2:
        raise ValueError(f"MXFP8 expects a 2D tensor, got shape={tuple(weight.shape)}")
    rows, cols = weight.shape
    if cols % MXFP8_GROUP_SIZE != 0:
        raise ValueError(
            f"MXFP8 requires the reduction dimension to be a multiple of {MXFP8_GROUP_SIZE}, got cols={cols}"
        )

    # TE's quantize kernel additionally requires rows % 32 == 0 (it asserts on
    # flat_first_dim); tensors that miss it take the torch path below.
    if weight.is_cuda and rows % MXFP8_GROUP_SIZE == 0:
        quantizer = _te_mx_quantizer()
        if quantizer is not None:
            return _te_mx_cast_2d(weight.detach().contiguous(), quantizer)

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    groups = weight.detach().to(torch.float32).contiguous().view(rows, cols // MXFP8_GROUP_SIZE, MXFP8_GROUP_SIZE)
    exponent = _mx_shared_exponent(groups.abs().amax(dim=-1))
    scale = torch.pow(2.0, exponent)

    codes = (groups / scale[..., None]).clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn)
    scales = (exponent + E8M0_BIAS).to(torch.uint8)
    return codes.view(rows, cols).contiguous(), scales.contiguous()


def batched_mx_cast_to_fp8(
    weight: torch.Tensor,
    expert_batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D ``[experts, rows, cols]`` tensor to MXFP8.

    Batches experts for the same reason the blockwise path does: one conversion
    pipeline per batch instead of per expert, with bounded FP32 workspace.
    """

    if weight.ndim != 3:
        raise ValueError(f"Batched MXFP8 expects a 3D tensor, got shape={tuple(weight.shape)}")
    if isinstance(expert_batch_size, bool) or not isinstance(expert_batch_size, int) or expert_batch_size <= 0:
        raise ValueError(f"expert_batch_size must be a positive integer, got {expert_batch_size!r}")

    num_experts, rows, cols = weight.shape
    if cols % MXFP8_GROUP_SIZE != 0:
        raise ValueError(
            f"MXFP8 requires the reduction dimension to be a multiple of {MXFP8_GROUP_SIZE}, got cols={cols}"
        )

    if weight.is_cuda and num_experts * rows > 0 and (num_experts * rows) % MXFP8_GROUP_SIZE == 0:
        quantizer = _te_mx_quantizer()
        if quantizer is not None:
            # MX groups run along the last dim, so experts and rows flatten
            # without crossing any group boundary: one kernel over [E*N, K]
            # instead of a python loop of expert batches.
            flat = weight.detach().reshape(num_experts * rows, cols).contiguous()
            codes, scales = _te_mx_cast_2d(flat, quantizer)
            return (
                codes.view(num_experts, rows, cols),
                scales.view(num_experts, rows, cols // MXFP8_GROUP_SIZE),
            )

    num_groups = cols // MXFP8_GROUP_SIZE
    codes = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty((num_experts, rows, num_groups), dtype=torch.uint8, device=weight.device)

    for start in range(0, num_experts, expert_batch_size):
        end = min(start + expert_batch_size, num_experts)
        # MX groups run along the last dim, so a batch of experts flattens to
        # 2D without crossing group boundaries; the 2D cast is the single
        # owner of the quantization rule.
        batch_codes, batch_scales = mx_cast_to_fp8(weight[start:end].detach().reshape(-1, cols))
        codes[start:end].copy_(batch_codes.view(end - start, rows, cols))
        scales[start:end].copy_(batch_scales.view(end - start, rows, num_groups))

    return codes, scales
