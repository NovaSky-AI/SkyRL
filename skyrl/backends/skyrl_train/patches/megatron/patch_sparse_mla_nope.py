"""Backport of NVIDIA/Megatron-LM#7617: NoPE shapes and unaligned top-k in TileLang SparseMLA.

megatron-core's fused absorbed sparse-attention hook (``tilelang_dsa.fused_sparse_mla_absorbed``)
only accepts the DeepSeek-V3.2 / GLM-5 absorbed layout: q/k width 576 (512 latent + 64 RoPE) and a
top-k width that is a multiple of 64. GLM-5.3-Flash's DSA layers are NoPE MLA (q/k width 512), and
its k-pool selection appends the query's incomplete tail pool (2048 + pool_size - 1 = 2051 slots),
so the hook declines and ``DSAttention`` falls back to a dense ``[b, heads, sq, sk]`` FP32 masked
softmax -- O(L^2) memory, ~128 GiB/GPU at 32k tokens.

Upstream zero-pads 512-wide q/k to 576 (zero channels add nothing to q.k; the caller's
``softmax_scale`` is kept, the value is still ``key[..., :512]``) and pads the indices to a multiple
of 64 with -1, which the kernel masks. Both are exact. ``_fused_sparse_mla_absorbed`` below is the
upstream function from the PR head (3d073471), with the module globals it reads (``SparseMLA``,
``_all_bfloat16``, ``_is_supported_sparse_mla_head_count``) looked up on ``tilelang_dsa`` at call
time instead.

Its only caller is ``tilelang_dsa.run_fused_absorbed_sparse_attention``, which resolves it as a
module global, so rebinding the attribute on ``tilelang_dsa`` is enough. Only active with
``dsa_kernel_backend="tilelang"``; the 576-wide, 64-aligned path is unchanged.

DELETE THIS PATCH once the megatron-core pin includes NVIDIA/Megatron-LM#7617.
"""

import inspect
from typing import Optional

import torch
from loguru import logger

_APPLIED = False

# Present in upstream's ``fused_sparse_mla_absorbed`` once NVIDIA/Megatron-LM#7617 lands.
_UPSTREAM_SENTINEL = "query.size(-1) not in (512, 576)"


def _fused_sparse_mla_absorbed(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
) -> Optional[torch.Tensor]:
    """Run fused SparseMLA kernel for absorbed-MLA path."""
    from megatron.core.transformer.experimental_attention_variant.ops import (
        tilelang_dsa,
    )

    if tilelang_dsa.SparseMLA is None:
        return None

    if query.ndim != 4 or key.ndim != 4 or topk_indices.ndim != 3:
        return None
    if not tilelang_dsa._all_bfloat16(query, key):
        return None
    if key.size(2) != 1:
        return None
    if query.size(1) != key.size(1) or topk_indices.size(0) != query.size(1):
        return None
    if topk_indices.size(1) != query.size(0):
        return None
    if query.size(-1) != key.size(-1):
        return None
    if query.size(-1) not in (512, 576) or v_channels != 512:
        # Current copied TileLang kernels are specialized for GLM5/DeepSeek V3.2 absorbed dims.
        return None
    query_heads = query.size(2)
    if query_heads <= 0:
        return None
    kernel_heads = max(query_heads, 16)
    if not tilelang_dsa._is_supported_sparse_mla_head_count(kernel_heads, kv_group=key.size(2)):
        return None
    if query.size(-1) == 512:
        # NoPE has no positional channels. Zero padding preserves QK scores and gradients
        # while satisfying SparseMLA's 512 latent + 64 RoPE layout; keep the caller's scale.
        query = torch.nn.functional.pad(query, (0, 64))
        key = torch.nn.functional.pad(key, (0, 64))
    if topk_indices.size(-1) % 64 != 0:
        # Invalid slots leave the selected keys unchanged, including KPool tail tokens.
        topk_indices = torch.nn.functional.pad(topk_indices, (0, -topk_indices.size(-1) % 64), value=-1)

    query_bshd = query.permute(1, 0, 2, 3).contiguous()
    if kernel_heads != query_heads:
        # SparseMLA uses a minimum 16-head tile without head bounds. Pad the caller
        # tensor so small TP shards stay in bounds, then discard those heads below.
        query_bshd = torch.nn.functional.pad(query_bshd, (0, 0, 0, kernel_heads - query_heads))
    key_bshd = key.permute(1, 0, 2, 3).contiguous()
    indices_bsgk = topk_indices.unsqueeze(2).to(torch.int32).contiguous()
    out, _ = tilelang_dsa.SparseMLA.apply(query_bshd, key_bshd, indices_bsgk, softmax_scale)
    if out.ndim != 4 or out.size(2) != kernel_heads or out.size(-1) != v_channels:
        return None
    out = out[:, :, :query_heads]
    return out.permute(1, 0, 2, 3).contiguous()


def patch_sparse_mla_nope() -> bool:
    """Rebind ``tilelang_dsa.fused_sparse_mla_absorbed`` to the #7617 version.

    Returns True if the patch is in place, False if megatron-core already has the fix or the
    TileLang DSA module is unavailable.
    """
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from megatron.core.transformer.experimental_attention_variant.ops import (
            tilelang_dsa,
        )
    except ImportError as e:
        logger.warning(f"megatron-core TileLang DSA module unavailable; skipping SparseMLA NoPE patch: {e}")
        return False

    if _UPSTREAM_SENTINEL in inspect.getsource(tilelang_dsa.fused_sparse_mla_absorbed):
        logger.warning(
            "megatron-core already supports NoPE SparseMLA (NVIDIA/Megatron-LM#7617); "
            "delete skyrl/backends/skyrl_train/patches/megatron/patch_sparse_mla_nope.py"
        )
        return False

    tilelang_dsa.fused_sparse_mla_absorbed = _fused_sparse_mla_absorbed
    _APPLIED = True
    logger.info("Applied TileLang SparseMLA NoPE / unaligned top-k patch (NVIDIA/Megatron-LM#7617)")
    return True
