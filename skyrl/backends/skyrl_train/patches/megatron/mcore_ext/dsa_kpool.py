"""DSA k-pool indexer kernels, vendored from NVIDIA/Megatron-LM#7522.

GLM-5.3-Flash's DSA layers use a *k-pool* compressed indexer: keys are pooled in groups of
``index_kpool`` consecutive tokens, the indexer scores and selects ``index_topk / index_kpool``
pools, the selected pools are expanded back to token indices, and the query's own incomplete
tail pool is always appended. Without it the token-level indexer is exact only up to
``dsa_indexer_topk`` (2048 for this checkpoint), which caps training at short sequences.

The pinned megatron-core does not have this yet -- NVIDIA/Megatron-LM#7522 ("Add KPool DSA
indexer, NoPE MLA support, and FP8 precision controls") is still open. The functions below are
copied **verbatim** from that PR's branch (HollowMan6/Megatron-LM @ beb4be3a8,
``megatron/core/transformer/experimental_attention_variant/dsa.py`` lines 743-993) so they can
be diffed against it mechanically::

    git diff <this file> <megatron>/core/transformer/experimental_attention_variant/dsa.py

Everything they depend on -- ``_compute_index_scores``, ``hadamard_transform``, ``dsa_masking``
-- already exists in the pinned megatron-core with identical signatures, so this module needs
no patching of megatron itself.

DELETE THIS MODULE once the megatron-core pin includes #7522.
"""

from typing import Optional, Tuple

import torch
from megatron.core.transformer.experimental_attention_variant import dsa_masking
from megatron.core.transformer.experimental_attention_variant.dsa import (
    _compute_index_scores,
)

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def _kpool_fp8_input(x: torch.Tensor) -> torch.Tensor:
    """Match the indexer's FP32 Hadamard, BF16 rounding, and E4M3 power-of-two scale."""
    if not x.numel():
        return x.float()
    assert hadamard_transform is not None, "fast_hadamard_transform is required for FP8 KPool."
    x = hadamard_transform(x.float(), scale=x.shape[-1] ** -0.5).to(torch.bfloat16).float()
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


def _kpool_compress_keys(k: torch.Tensor, gate_score: torch.Tensor, ape: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Softmax-weighted pool keys, accumulated in FP32 and returned in BF16.

    Keys and gates are [tokens, batch, head_dim]; ape is [pool_size, head_dim].
    Only complete pools are compressed. Query-local tails are appended separately.
    """
    seqlen, bsz, head_dim = k.shape
    assert head_dim == ape.shape[1], f"head_dim {head_dim} != ape dim1 {ape.shape[1]}"
    num_pools = seqlen // pool_size
    # Drop the trailing incomplete pool from compression; its tokens are appended
    # later via append_tail_to_topk (always_select_tail).
    usable = num_pools * pool_size
    # [num_pools, pool_size, batch, head_dim]
    k_p = k[:usable].reshape(num_pools, pool_size, bsz, head_dim)
    # gate_score: [seqlen, batch, head_dim] -> [num_pools, pool_size, batch, head_dim]
    if gate_score is not None:
        g = gate_score[:usable].reshape(num_pools, pool_size, bsz, head_dim).float()
    else:
        g = torch.zeros((num_pools, pool_size, bsz, head_dim), dtype=torch.float32, device=k.device)

    # Per-dim softmax across the pool's slots: score[slot] = gate_score[slot] + ape[slot].
    # ape: [pool_size, head_dim] -> broadcast over (num_pools, batch).
    ape_f = ape.to(dtype=torch.float32, device=k.device)  # [pool_size, head_dim]
    score = g + ape_f.unsqueeze(0).unsqueeze(2)  # [num_pools, pool_size, batch, head_dim]
    # Numerically-stable per-dim softmax over dim=1 (the pool slot dim).
    score_max = score.max(dim=1, keepdim=True).values
    prob = torch.exp(score - score_max)
    # weighted sum of k over pool slots: [num_pools, batch, head_dim]. Keep the
    # numerator 4D ([num_pools, 1, batch, head_dim]) so it broadcasts cleanly
    # against the 4D denom ([num_pools, 1, 1, head_dim]); a 3D numerator would
    # left-pad and produce a spurious extra (num_pools) dimension.
    k_f = k_p.float()
    k_pooled = (prob * k_f).sum(dim=1, keepdim=True) / prob.sum(dim=1, keepdim=True).clamp(min=1e-12)
    k_pooled = k_pooled.squeeze(1)  # [num_pools, batch, head_dim]
    return k_pooled.to(torch.bfloat16)


def _expand_pools_to_tokens(
    pool_ids: torch.Tensor,
    pool_valid: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
    pool_token_base: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand fixed-width pool IDs to token IDs, preserving -1 padding."""
    assert pool_ids.ndim == 2 and pool_valid.shape == pool_ids.shape
    assert pool_ids.shape[1] * pool_size == topk_tokens
    if pool_token_base is None:
        starts = pool_ids * pool_size
    elif pool_token_base.numel():
        starts = pool_token_base[pool_ids.clamp(min=0)]
    else:
        starts = torch.zeros_like(pool_ids)
    offsets = torch.arange(pool_size, device=pool_ids.device)
    tokens = starts.unsqueeze(-1) + offsets
    tokens = tokens.masked_fill(~pool_valid.unsqueeze(-1), -1)
    return tokens.reshape(pool_ids.shape[0], topk_tokens).to(torch.int32)


def _append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    tail_start_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Append each query's incomplete causal pool, in global token coordinates."""
    tail_count = seq_lens.to(torch.int32).remainder(pool_size)
    tail_start = (
        seq_lens.to(torch.int32) - tail_count if tail_start_override is None else tail_start_override.to(torch.int32)
    )
    offsets = torch.arange(pool_size - 1, device=topk_result.device)
    tail = tail_start[:, None] + offsets
    tail = tail.masked_fill(offsets >= tail_count[:, None], -1).to(topk_result.dtype)
    return torch.cat((topk_result, tail), dim=-1)


def _kpool_compress_keys_per_seg(
    k: torch.Tensor,
    gate_score: Optional[torch.Tensor],
    ape: torch.Tensor,
    pool_size: int,
    cu_seqlens_kv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress complete pools within each packed segment.

    Return pooled keys and their global starting token indices. Segment boundaries
    need not be multiples of pool_size; a pool must never span two documents.
    """
    cu = cu_seqlens_kv.to(device=k.device, dtype=torch.int64)
    n_seg = int(cu.numel()) - 1
    pooled_parts = []
    base_parts = []
    for i in range(n_seg):
        s = int(cu[i])
        e = int(cu[i + 1])
        seg_len = e - s
        if seg_len <= 0:
            continue
        k_seg = k[s:e]
        gate_seg = gate_score[s:e] if gate_score is not None else None
        k_pooled_seg = _kpool_compress_keys(k_seg, gate_seg, ape, pool_size)
        # [num_pools_seg, b, d]
        n_pools_seg = k_pooled_seg.size(0)
        pooled_parts.append(k_pooled_seg)
        # pool j (local) of this segment starts at global token s + j*pool_size.
        seg_bases = torch.arange(n_pools_seg, device=k.device, dtype=torch.int64) * pool_size + s
        base_parts.append(seg_bases)
    k_pooled_global = torch.cat(pooled_parts, dim=0)
    pool_token_base = torch.cat(base_parts, dim=0)
    return k_pooled_global, pool_token_base


def fused_qk_topk_kpool(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    pool_size: int,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    use_relu: bool = True,
    always_select_tail: bool = True,
    fp8_indexer: bool = False,
):
    """Select complete causal pools and append each query's incomplete tail.

    q is [queries, batch, heads, dim], k/gate_score are [keys, batch, dim],
    and weights are [queries, batch, heads]. Packed bounds use global token
    coordinates. Output indices are [batch, queries, index_topk + pool_size - 1]
    when always_select_tail is enabled, with -1 for unused slots.
    """
    sk = k.size(0)

    # Packed pools restart at each document boundary.
    use_per_seg = cu_seqlens_kv is not None and cu_seqlens_kv.numel() >= 2
    if use_per_seg:
        k_pooled, pool_token_base = _kpool_compress_keys_per_seg(k, gate_score, ape, pool_size, cu_seqlens_kv)
        num_pools = k_pooled.size(0)
    else:
        num_pools = sk // pool_size
        k_pooled = _kpool_compress_keys(k, gate_score, ape, pool_size)
        pool_token_base = torch.arange(num_pools, device=k.device, dtype=torch.int64) * pool_size

    if fp8_indexer:
        q, k_pooled = _kpool_fp8_input(q), _kpool_fp8_input(k_pooled)
    index_scores = _compute_index_scores(q, weights, k_pooled, use_relu=use_relu)

    # A pool is causal only when its final token is within the query's bounds.
    pool_positions = pool_token_base + (pool_size - 1)
    eff_key_positions = key_positions[pool_positions] if key_positions is not None else pool_positions
    v_starts, v_ends, k_pos = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=eff_key_positions,
        sk=num_pools,
        device=index_scores.device,
    )
    if v_starts is not None:
        index_scores = dsa_masking.apply_starts_ends_mask_to_scores(index_scores, v_starts, v_ends, k_pos)
    elif mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # Keep the selection width fixed, including when fewer causal pools exist.
    budget = index_topk // pool_size
    select_k = min(budget, num_pools)
    if select_k > 0:
        topk_scores, pool_topk = index_scores.topk(select_k, dim=-1)
        # [batch, seqlen_q, select_k] -> mask invalid pools
        pool_topk = pool_topk.masked_fill(topk_scores == float("-inf"), -1)
    else:
        pool_topk = torch.empty(index_scores.shape[:-1] + (0,), dtype=torch.int64, device=index_scores.device)
    if pool_topk.shape[-1] < budget:
        pad = torch.full(
            index_scores.shape[:-1] + (budget - pool_topk.shape[-1],),
            -1,
            dtype=torch.int64,
            device=index_scores.device,
        )
        pool_topk = torch.cat([pool_topk, pad], dim=-1)

    # Expand [batch * queries, pools] to a fixed token budget.
    rows = pool_topk.shape[0] * pool_topk.shape[1]
    pool_flat = pool_topk.reshape(rows, -1)
    pool_valid = pool_flat >= 0
    # Clamp invalid ids to 0 for the arithmetic, restore -1 via the where mask.
    safe_pool = pool_flat.clamp(min=0)
    token_topk = _expand_pools_to_tokens(
        safe_pool,
        pool_valid,
        index_topk,
        pool_size,
        pool_token_base=pool_token_base if use_per_seg else None,
    )
    # token_topk is [rows, index_topk]; reshape back to [batch, seqlen_q, index_topk].
    token_topk = token_topk.reshape(pool_topk.shape[0], pool_topk.shape[1], index_topk)

    if always_select_tail:
        # Pool phase is query-local, never the final length of the packed sample.
        sq, batch = q.shape[:2]
        ends = v_ends if v_ends is not None else torch.arange(1, sq + 1, device=q.device)
        if v_starts is not None:
            starts = v_starts
        elif cu_seqlens_kv is not None:
            cu = cu_seqlens_kv.to(device=q.device, dtype=torch.int64)
            starts = cu[torch.searchsorted(cu[1:], ends - 1, right=True)]
        else:
            starts = torch.zeros_like(ends)
        lengths = ends - starts
        tail_starts = ends - lengths.remainder(pool_size)
        token_topk = _append_tail_to_topk(
            token_topk.reshape(rows, -1),
            lengths.expand(batch, -1).reshape(-1),
            pool_size,
            tail_start_override=tail_starts.expand(batch, -1).reshape(-1),
        ).reshape(batch, sq, -1)

    return index_scores, token_topk
