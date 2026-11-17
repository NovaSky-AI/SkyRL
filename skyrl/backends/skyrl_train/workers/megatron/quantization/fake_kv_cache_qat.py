"""Fake-quantized KV cache QAT for absorbed-MLA (DSA) attention: the trainer attends over the rollouts' cache rows.

vLLM's sparse-MLA backends (FlashMLA sparse, DeepSeek-V3.2 / GLM-5.x geometry) read every key from the KV cache,
including the current prefill chunk's own tokens, and dequantize it to bf16 before bf16 attention math with a bf16
query. So with a quantized cache, every key and value the rollouts attend over is ``dequant(cache_row)``. The cache row
is exactly megatron-core's absorbed key, ``[kv_layernorm(latent) | rope(k_pe)]`` (``AbsorbedMLASelfAttention``), whose
first ``kv_lora_rank`` dims ``DSAttention`` also uses as the value.

This module replaces that key with the dequantized cache row vLLM would hold, computed in one pass by
:func:`skyrl.backends.skyrl_train.kernels.kv_cache_round_trip.kv_cache_round_trip`: vLLM's own RMSNorm and RoPE helpers
(fp32, rounded straight to the cache format, never through bf16) and the writer's rounding, decoded exactly and stored
in megatron's key column order. It is bit-identical to vLLM's cache writer (``fused_norm_rope``) followed by its reader
(``cp_gather_and_upconvert_*_kv_cache``), which :func:`cache_round_trip` runs as the reference. Its inputs are the
pre-norm latent and raw ``k_pe`` (captured from ``linear_kv_down_proj``), the live ``kv_layernorm`` weight, each row's
position in its sequence, and vLLM's bf16 cos/sin table.

Backward: everything downstream of the cache sees the round trip (attention saves the rounded key, so dQ, dP and dK,
and the up-projection weight gradients, use it); the rounding itself is straight-through, so the key's gradient reaches
megatron's own norm and RoPE backward unchanged. Nothing extra is saved for backward.

Formats: ``nvfp4_ds_mla`` (E2M1 latent with E4M3 scales per 16, E4M3 RoPE) and ``fp8_ds_mla`` (E4M3 latent with fp32
scales per 128, bf16 RoPE).
"""

from __future__ import annotations

import functools
from enum import StrEnum
from types import SimpleNamespace
from typing import Any, Callable, Optional

import torch
from loguru import logger


class KvCacheFormat(StrEnum):
    """vLLM ``kv_cache_dtype`` values this module emulates."""

    NVFP4_DS_MLA = "nvfp4_ds_mla"
    FP8_DS_MLA = "fp8_ds_mla"


_CACHE_ROW_BYTES = {KvCacheFormat.NVFP4_DS_MLA: 352, KvCacheFormat.FP8_DS_MLA: 656}
_KV_LORA_RANK = 512
_ROPE_DIM = 64
_SCRATCH_BLOCK = 64
# Positions above this are clamped (only padding rows can reach it); the bf16 table is 128 MiB per device.
MAX_POSITIONS = 1 << 20
# A wrong position, rope base or dim order moves the RoPE half by O(1); the cache's own rounding (E4M3 for nvfp4_ds_mla)
# and vLLM's bf16 cos/sin vs megatron's fp32 ones move it by ~3e-2 (GLM-5.3).
_POSITION_CHECK_TOL = 0.1

_installed_format: Optional[KvCacheFormat] = None
_position_check_pending = True


@functools.cache
def _cos_sin_cache(device: torch.device, rope_theta: float) -> torch.Tensor:
    """vLLM's ``RotaryEmbedding`` cos/sin table (default rope type), computed as vLLM does: on the GPU, then bf16."""
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    rope = SimpleNamespace(base=rope_theta, rotary_dim=_ROPE_DIM, max_position_embeddings=MAX_POSITIONS)
    rope._compute_inv_freq = functools.partial(RotaryEmbedding._compute_inv_freq, rope)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.device(device):
            return RotaryEmbedding._compute_cos_sin_cache(rope).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


@functools.cache
def _megatron_rope_order(device: torch.device) -> torch.Tensor:
    """Column order of megatron's absorbed key: its MLA RoPE (``mla_rotary_interleaved``) rotates the pairs
    ``(x[2i], x[2i+1])`` like vLLM but stores them deinterleaved, ``[x[0::2] | x[1::2]]``; vLLM's cache keeps them
    interleaved. q is permuted the same way, so the scores agree."""
    rope = torch.arange(_KV_LORA_RANK, _KV_LORA_RANK + _ROPE_DIM, device=device)
    return torch.cat([torch.arange(_KV_LORA_RANK, device=device), rope[0::2], rope[1::2]])


def to_megatron_key_order(rows: torch.Tensor) -> torch.Tensor:
    """vLLM cache rows ``[T, 576]`` -> megatron's absorbed-key column order."""
    return rows.index_select(-1, _megatron_rope_order(rows.device))


@functools.cache
def _dummies(device: torch.device) -> SimpleNamespace:
    """Inputs of ``fused_norm_rope``'s query and indexer programs, which this module does not use."""
    return SimpleNamespace(
        topk=torch.empty((1, 1), dtype=torch.int32, device=device),
        q_norm_w=torch.ones((1,), dtype=torch.bfloat16, device=device),
        workspace_starts=torch.zeros((1,), dtype=torch.int32, device=device),
    )


def cache_round_trip(
    kv_combined: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rope_theta: float,
    fmt: KvCacheFormat,
    k_pe_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference: ``[T, 576]`` bf16 dequantized vLLM cache rows of pre-norm ``kv_combined [T, 512 + 64]`` (the
    ``kv_a_proj`` output) at ``positions [T]``, by vLLM's own cache writer and reader, in vLLM's column order (see
    :func:`to_megatron_key_order`). ``k_pe_out [T, 64]``, if given, receives vLLM's bf16 RoPE'd ``k_pe``."""
    from vllm import _custom_ops as ops
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope

    rows, device = kv_combined.size(0), kv_combined.device
    out = torch.empty((rows, _KV_LORA_RANK + _ROPE_DIM), dtype=torch.bfloat16, device=device)
    if rows == 0:
        return out
    blocks = -(-rows // _SCRATCH_BLOCK)
    cache = torch.empty((blocks, _SCRATCH_BLOCK, _CACHE_ROW_BYTES[fmt]), dtype=torch.uint8, device=device)
    d = _dummies(device)
    # The query program normalizes a 1-wide dummy per row.
    q = torch.empty((rows, 1), dtype=torch.bfloat16, device=device)
    fused_norm_rope(
        positions,
        q,
        d.q_norm_w,
        eps,
        kv_combined[:, :_KV_LORA_RANK],
        norm_weight,
        eps,
        kv_combined[:, _KV_LORA_RANK:],
        _cos_sin_cache(device, rope_theta),
        None,
        None,
        None,
        eps,
        None,
        d.topk,
        slot_mapping=torch.arange(rows, dtype=torch.int64, device=device),
        mla_kv_cache=cache,
        mla_kv_cache_dtype=fmt.value,
        has_indexer=False,
        q_c_out=q,
        k_pe_out=k_pe_out,
    )
    block_table = torch.arange(blocks, dtype=torch.int32, device=device).view(1, blocks)
    if fmt == KvCacheFormat.NVFP4_DS_MLA:
        ops.cp_gather_and_upconvert_nvfp4_kv_cache(cache, out, block_table, d.workspace_starts, 1)
    else:
        ops.cp_gather_and_upconvert_fp8_kv_cache(cache, out, block_table, d.workspace_starts, 1)
    return out


class _CacheRowStraightThrough(torch.autograd.Function):
    """Forward: the dequantized cache rows; backward: the key's gradient unchanged."""

    @staticmethod
    def forward(ctx, key: torch.Tensor, round_trip: Callable[[], torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return round_trip().view(key.shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


def row_positions(
    packed_seq_params: Any, seq_len: int, batch: int, cp_size: int, cp_rank: int, device: torch.device
) -> torch.Tensor:
    """Each local key row's position in its sequence (vLLM's ``positions``), in the row order of the absorbed key:
    packed THD rows under megatron's per-sequence zigzag CP split, or ``[s, b]`` rows under the plain zigzag split."""
    from megatron.core.transformer.experimental_attention_variant import dsa_layout

    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        cu = cu.to(device=device, dtype=torch.int64)
        packed = dsa_layout.build_packed_allgather_cp_local_positions(
            cu, cp_size, cp_rank, device, output_size=seq_len * batch
        )
        seq = (torch.searchsorted(cu, packed, right=True) - 1).clamp(0, cu.numel() - 2)
        pos = packed - cu.index_select(0, seq)
    else:
        local = dsa_layout.build_zigzag_cp_local_positions(seq_len * cp_size, cp_size, cp_rank, device)
        pos = local.view(seq_len, 1).expand(seq_len, batch).reshape(-1)
    return pos.clamp(0, MAX_POSITIONS - 1)


def _check_positions(key: torch.Tensor, rows: torch.Tensor) -> None:
    """The trainer's own RoPE'd ``k_pe`` must match the cache rows' at the derived positions (once per process)."""
    global _position_check_pending
    _position_check_pending = False
    mine = key.reshape(-1, key.size(-1))[:, _KV_LORA_RANK:].float()
    rel = (
        (mine - rows.reshape(mine.size(0), -1)[:, _KV_LORA_RANK:].float()).norm() / mine.norm().clamp_min(1e-12)
    ).item()
    if rel > _POSITION_CHECK_TOL:
        raise RuntimeError(
            f"KV cache QAT: the trainer's RoPE'd k_pe differs from the cache rows' at the derived positions (rel "
            f"{rel:.3g}); the row positions or the rope base are wrong for this layout"
        )
    logger.info(f"KV cache QAT: trainer vs cache-row RoPE'd k_pe agree at the derived positions (rel {rel:.2e}).")


def _wrap_get_query_key_value_tensors(orig: Callable, fmt: KvCacheFormat) -> Callable:
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
    )

    from skyrl.backends.skyrl_train.kernels.kv_cache_round_trip import (
        kv_cache_round_trip,
    )

    @functools.wraps(orig)
    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        if inference_context is not None or inference_params is not None:
            return orig(
                self,
                hidden_states,
                key_value_states,
                packed_seq_params,
                inference_context,
                inference_params=inference_params,
            )
        stash = []
        handle = self.linear_kv_down_proj.register_forward_hook(lambda _m, _i, out: stash.append(out[0]))
        try:
            query, key, q_compressed = orig(self, hidden_states, key_value_states, packed_seq_params)
        finally:
            handle.remove()
        cfg = self.config
        kv = stash[-1].detach()
        if kv.size(-1) != cfg.kv_lora_rank + cfg.qk_pos_emb_head_dim:
            raise NotImplementedError("KV cache QAT needs an unsharded linear_kv_down_proj output")
        if torch.distributed.get_world_size(self.tp_group) > 1 and cfg.sequence_parallel:
            kv = gather_from_sequence_parallel_region(kv, group=self.tp_group)
        seq_len, batch = (kv.size(0), 1) if kv.dim() == 2 else (kv.size(0), kv.size(1))
        kv = kv.reshape(-1, kv.size(-1))
        cp = self.pg_collection.cp
        positions = row_positions(packed_seq_params, seq_len, batch, cp.size(), cp.rank(), kv.device)
        norm_weight = self.kv_layernorm.weight.detach()
        cos_sin = _cos_sin_cache(kv.device, cfg.rotary_base)
        nvfp4 = fmt == KvCacheFormat.NVFP4_DS_MLA
        rows = _CacheRowStraightThrough.apply(
            key, lambda: kv_cache_round_trip(kv, positions, norm_weight, cfg.layernorm_epsilon, cos_sin, nvfp4)
        )
        if _position_check_pending:
            _check_positions(key.detach(), rows.detach())
        return query, rows, q_compressed

    get_query_key_value_tensors.__wrapped__ = orig
    return get_query_key_value_tensors


def check_kv_cache_qat_config(config: Any) -> None:
    """The geometry and RoPE the emulated vLLM cache formats assume (DeepSeek-V3.2 / GLM-5.x)."""
    problems = []
    if config.kv_lora_rank != _KV_LORA_RANK or config.qk_pos_emb_head_dim != _ROPE_DIM:
        problems.append(
            f"kv_lora_rank/qk_pos_emb_head_dim {config.kv_lora_rank}/{config.qk_pos_emb_head_dim} != 512/64"
        )
    if config.layernorm_zero_centered_gamma:
        problems.append("layernorm_zero_centered_gamma")
    if config.rotary_scaling_factor != 1.0:
        problems.append(f"rotary_scaling_factor {config.rotary_scaling_factor} (vLLM's default rope type has none)")
    if problems:
        raise ValueError(f"KV cache QAT does not support this attention: {', '.join(problems)}")


def install_kv_cache_qat(fmt: KvCacheFormat | str) -> None:
    """Make every ``AbsorbedMLASelfAttention`` in this process attend over ``fmt`` cache rows, once per process."""
    global _installed_format
    fmt = KvCacheFormat(fmt)
    if _installed_format is not None:
        if _installed_format != fmt:
            raise ValueError(f"KV cache QAT already installed for {_installed_format}")
        return
    from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
        AbsorbedMLASelfAttention,
    )

    AbsorbedMLASelfAttention.get_query_key_value_tensors = _wrap_get_query_key_value_tensors(
        AbsorbedMLASelfAttention.get_query_key_value_tensors, fmt
    )
    _installed_format = fmt
    logger.info(f"KV cache QAT: absorbed-MLA keys are the dequantized vLLM {fmt.value} cache rows.")


def absorbed_mla_attentions(model: Any) -> list:
    """The ``AbsorbedMLASelfAttention`` modules of a (list of) Megatron model chunk(s)."""
    from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
        AbsorbedMLASelfAttention,
    )

    chunks = model if isinstance(model, (list, tuple)) else [model]
    return [m for chunk in chunks for m in chunk.modules() if isinstance(m, AbsorbedMLASelfAttention)]
