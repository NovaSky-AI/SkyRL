"""Grouped NVFP4 x NVFP4 GEMM for routed-expert fprop on Blackwell tensor cores.

``out[r] = alpha[e] * pts[r] * sum_k (e2m1 * sf)_x[r, k] * (e2m1 * sf)_w[e][n, k]`` for row ``r`` of
expert ``e``, accumulated in fp32 and rounded to bf16 once: the operands are exactly the expert_nvfp4
round trips (:func:`.per_token.pack_per_token`, :func:`.fake_experts.pack_expert_weights`), without
rounding their dequantized values to bf16, as vLLM's FlashInfer TRT-LLM W4A4 MoE computes it.

Runs FlashInfer's CuTe DSL contiguous grouped GEMM (``blockscaled_contiguous_grouped_gemm_finalize_fusion``)
with the finalize scatter off and unit routing scales: a plain grouped GEMM over per-expert row ranges
padded to the MMA tile, writing unpadded rows. Tile metadata is built on the device from the host
``m_splits`` TE already holds, so no host sync.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

# MMA tile (M, N) and cluster shape: the fastest tactic on B200 for GLM-5.3 expert shapes.
MMA_TILER_MN = (256, 256)
CLUSTER_SHAPE_MN = (2, 1)
ROW_TILE = MMA_TILER_MN[0]


@dataclass
class TileLayout:
    """Per-expert row ranges padded to ``ROW_TILE``: ``dst`` maps each unpadded row to its padded row,
    ``perm`` the reverse (``-1`` for padding), and the per-tile expert / row-limit tables."""

    rows: int
    padded_rows: int
    dst: torch.Tensor  # (rows,) int32
    perm: torch.Tensor  # (padded_rows,) int32
    tile_expert: torch.Tensor  # (tiles,) int32
    tile_row_limit: torch.Tensor  # (tiles,) int32
    num_tiles: torch.Tensor  # (1,) int32
    unit_scales: torch.Tensor  # (rows, 1) fp32 routing scales of 1


def tile_layout(m_splits: Sequence[int], device: torch.device) -> TileLayout:
    tiles = [-(-m // ROW_TILE) for m in m_splits]
    rows, num_tiles = sum(m_splits), sum(tiles)
    padded_starts = _cumsum([t * ROW_TILE for t in tiles])
    tile_expert, tile_row_limit = [], []
    for e, (m, t) in enumerate(zip(m_splits, tiles)):
        for i in range(t):
            tile_expert.append(e)
            tile_row_limit.append(padded_starts[e] + min((i + 1) * ROW_TILE, m))
    # One host-to-device copy of the tables; dst[r] = r + (padded - unpadded start of r's expert).
    host = torch.tensor(
        tile_expert
        + tile_row_limit
        + [num_tiles]
        + [p - u for p, u in zip(padded_starts, _cumsum(m_splits))]
        + list(m_splits),
        dtype=torch.int32,
    ).to(device, non_blocking=True)
    e = len(m_splits)
    shift, counts = host[2 * num_tiles + 1 : 2 * num_tiles + 1 + e], host[2 * num_tiles + 1 + e :]
    dst = (
        torch.arange(rows, device=device, dtype=torch.int32)
        + torch.repeat_interleave(shift, counts.long(), output_size=rows)
    ).to(torch.int32)
    perm = torch.full((padded_rows := num_tiles * ROW_TILE,), -1, dtype=torch.int32, device=device)
    perm[dst.long()] = torch.arange(rows, dtype=torch.int32, device=device)
    return TileLayout(
        rows=rows,
        padded_rows=padded_rows,
        dst=dst,
        perm=perm,
        tile_expert=host[:num_tiles],
        tile_row_limit=host[num_tiles : 2 * num_tiles],
        num_tiles=host[2 * num_tiles : 2 * num_tiles + 1],
        unit_scales=torch.ones((rows, 1), dtype=torch.float32, device=device),
    )


def _cumsum(values: Sequence[int]) -> list[int]:
    out, acc = [], 0
    for v in values:
        out.append(acc)
        acc += v
    return out


def fp4_grouped_gemm(
    x_packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    w_packed,
    layout: TileLayout,
    out: torch.Tensor,
    out_dtype: str = "bfloat16",
) -> torch.Tensor:
    """``out (rows, N)`` = the grouped NVFP4 GEMM of packed per-token activations ``(codes, sf, pts)`` in ``layout``
    and packed expert weights (``fake_experts.PackedExperts``), in ``out_dtype`` (``"bfloat16"`` or ``"float32"``)."""
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        blockscaled_contiguous_grouped_gemm_finalize_fusion,
    )

    xq, xsf, pts = x_packed
    if layout.rows == 0:
        return out
    blockscaled_contiguous_grouped_gemm_finalize_fusion(
        a=xq,
        b=w_packed.q,
        a_scale=xsf,
        b_scale=w_packed.sf,
        alpha=w_packed.alpha,
        # Megatron's buffer holds a grouped linear's experts highest first: the GEMM reads them where they are.
        tile_idx_to_expert_idx=w_packed.gemm_expert_index(layout.tile_expert).to(torch.int32),
        num_non_exiting_tiles=layout.num_tiles,
        tile_idx_to_mn_limit=layout.tile_row_limit,
        permuted_idx_to_expanded_idx=layout.perm,
        token_final_scales=layout.unit_scales,
        out=out,
        a_per_token_scale=pts,
        a_dtype="float4_e2m1fn",
        b_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        out_dtype=out_dtype,
        sf_vec_size=16,
        mma_tiler_mn=MMA_TILER_MN,
        cluster_shape_mn=CLUSTER_SHAPE_MN,
        use_fused_finalize=False,
    )
    return out
