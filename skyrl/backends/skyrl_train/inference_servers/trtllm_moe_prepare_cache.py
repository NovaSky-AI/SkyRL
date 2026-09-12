"""Cache the fixed row permutation inside vLLM's MXFP8 TRT-LLM MoE prepare.

Every weight sync re-runs ``process_weights_after_loading`` on each FusedMoE
layer, and for compressed-tensors MXFP8 on the TRT-LLM backend that calls
``_shuffle_mxfp8_moe_weights`` (vllm .../utils/flashinfer_utils.py): a python
loop over all experts issuing per-expert ``reorder_rows_for_gated_act_gemm`` +
``shuffle_matrix_a`` + ``shuffle_matrix_sf_a`` kernels. Measured on
Qwen3.5-35B-A3B that is ~6.5 s of the sync's 8.1 s per engine, re-derived from
scratch on every sync even though the transform is a fixed relocation for
fixed shapes.

This module wraps the function with a learned row-permutation cache:

* All four tensors have 1-byte elements (E4M3 weights viewed as bytes, uint8
  scales), and the transform relocates whole rows (gate/up interleave and the
  epilogue-tile shuffle permute rows; ``torch.stack`` concatenates), so the
  entire function collapses to ``flat_rows.index_select(0, perm)`` per output.
* The permutation is learned empirically on first use per shape key: each row
  of a probe tensor is tagged with its row index (4 little-endian bytes,
  tiled), the original function is called once, and the tag of every output
  row identifies its source row.
* The learned permutation must then reproduce the original bitwise on two
  validation inputs (a fresh random tensor and the first real call's own
  output) before it is trusted. Any mismatch — sub-row movement, padding,
  arithmetic, a future vLLM layout change — permanently falls back to the
  original function for that key. Correctness never depends on this cache
  being right; only speed does.

Install from the worker extension (same pattern as ``patch_numel_loaded``).
``SKYRL_TRTLLM_MOE_PREPARE_CACHE=0`` disables it.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL = None
# key -> list of 4 row-permutation index tensors, or None (validated fallback)
_CACHE: dict[tuple, Optional[list[torch.Tensor]]] = {}


def _key(w13, w2, w13_scale, w2_scale, is_gated):
    return (
        tuple(w13.shape),
        tuple(w2.shape),
        tuple(w13_scale.shape),
        tuple(w2_scale.shape),
        w13.dtype,
        w2.dtype,
        w13_scale.dtype,
        w2_scale.dtype,
        bool(is_gated),
        w13.device.index,
    )


# A chunk permutation index is int64; cap its memory so the huge weight
# tensors must be captured at row granularity or not at all.
_MAX_INDEX_BYTES = 64 * 2**20


def _bytes(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


def _tag_chunks_like(t: torch.Tensor, chunk: int) -> torch.Tensor:
    """A probe whose every `chunk`-byte block is tiled with its block index (LE32)."""
    flat = _bytes(t).reshape(-1)
    n_chunks = flat.numel() // chunk
    idx = torch.arange(n_chunks, dtype=torch.int32, device=t.device)
    tags = (
        idx.unsqueeze(1)
        .expand(n_chunks, (chunk + 3) // 4)
        .contiguous()
        .view(torch.uint8)
        .reshape(n_chunks, -1)[:, :chunk]
    )
    return tags.reshape(_bytes(t).shape).view(t.dtype)


def _read_chunk_sources(out: torch.Tensor, chunk: int, n_chunks_in: int) -> Optional[torch.Tensor]:
    """Recover, for each output chunk, the input chunk it was copied from."""
    flat = _bytes(out).reshape(-1)
    if flat.numel() % chunk != 0:
        return None
    blocks = flat.reshape(-1, chunk)
    src = blocks[:, :4].contiguous().view(torch.int32).reshape(-1).to(torch.long)
    if src.numel() == 0 or src.min().item() < 0 or src.max().item() >= n_chunks_in:
        return None
    # a permutation duplicates nothing: every tag must be internally consistent
    if chunk >= 8:
        second = blocks[:, 4:8].contiguous().view(torch.int32).reshape(-1).to(torch.long)
        if not torch.equal(src, second):
            return None
    return src


def _apply_one(perm: torch.Tensor, chunk: int, t: torch.Tensor) -> torch.Tensor:
    blocks = _bytes(t).reshape(-1, chunk)
    out = blocks.index_select(0, perm)
    return out.reshape(_bytes(t).shape).view(t.dtype)


def _apply(entry: list[tuple[torch.Tensor, int]], tensors: tuple) -> tuple:
    return tuple(_apply_one(perm, chunk, t) for (perm, chunk), t in zip(entry, tensors))


def _learn(original, w13, w2, w13_scale, w2_scale, is_gated):
    """Learn a (perm, chunk) per output, coarsest granularity first.

    Returns ``(entry, ref_real)`` or ``None``. Row granularity is tried first
    (index ~8 B/row); 4-byte chunks are the fallback for transforms that move
    sub-row groups, allowed only while the index stays under _MAX_INDEX_BYTES.
    """

    inputs = (w13, w2, w13_scale, w2_scale)
    try:
        candidates: list[list[int]] = []
        for t in inputs:
            b = _bytes(t)
            options = []
            row = b.shape[-1]
            for chunk in (row, 4):
                if chunk < 4 or b.numel() % chunk:
                    continue
                n = b.numel() // chunk
                if n * 8 > _MAX_INDEX_BYTES:
                    continue
                options.append(chunk)
            if not options:
                return None
            candidates.append(options)

        # one probe call per distinct granularity attempt, coarsest first
        entry: list[Optional[tuple[torch.Tensor, int]]] = [None] * 4
        for attempt in range(max(len(o) for o in candidates)):
            chunks = [o[min(attempt, len(o) - 1)] for o in candidates]
            probes = tuple(_tag_chunks_like(t, c) for t, c in zip(inputs, chunks))
            probe_outs = original(*probes, is_gated)
            for i, (t, c, out) in enumerate(zip(inputs, chunks, probe_outs)):
                if entry[i] is not None:
                    continue
                if _bytes(out).numel() != _bytes(t).numel():
                    return None
                perm = _read_chunk_sources(out, c, _bytes(t).numel() // c)
                if perm is not None:
                    entry[i] = (perm, c)
            if all(e is not None for e in entry):
                break
        if not all(e is not None for e in entry):
            return None

        # Validation 1: a fresh random input must reproduce bitwise.
        rand_in = tuple(
            torch.randint(0, 256, _bytes(t).shape, dtype=torch.uint8, device=t.device).view(t.dtype) for t in inputs
        )
        ref = original(*rand_in, is_gated)
        got = _apply(entry, rand_in)
        for r, g in zip(ref, got):
            if not torch.equal(_bytes(r), _bytes(g)):
                return None

        # Validation 2: the caller's actual tensors must also reproduce bitwise.
        ref_real = original(*inputs, is_gated)
        got_real = _apply(entry, inputs)
        for r, g in zip(ref_real, got_real):
            if not torch.equal(_bytes(r), _bytes(g)):
                return None
        return entry, ref_real
    except Exception:
        logger.exception("TRTLLM MoE prepare cache: learning failed; falling back")
        return None


def install() -> bool:
    """Wrap vLLM's _shuffle_mxfp8_moe_weights with the permutation cache."""

    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return True
    if os.environ.get("SKYRL_TRTLLM_MOE_PREPARE_CACHE", "1") in ("0", "false", "False"):
        return False
    try:
        from vllm.model_executor.layers.quantization.utils import flashinfer_utils
    except Exception:
        return False
    original = getattr(flashinfer_utils, "_shuffle_mxfp8_moe_weights", None)
    if original is None:
        return False
    _ORIGINAL = original

    def cached_shuffle(*args, **kwargs):
        # Only the exact call shape this cache was built for takes the fast
        # path; any signature drift in vLLM's private helper delegates
        # verbatim instead of failing the sync on an argument-binding error.
        if kwargs or len(args) != 5:
            return original(*args, **kwargs)
        w13, w2, w13_scale, w2_scale, is_gated = args
        key = _key(w13, w2, w13_scale, w2_scale, is_gated)
        entry = _CACHE.get(key, "miss")
        if entry == "miss":
            learned = _learn(original, w13, w2, w13_scale, w2_scale, is_gated)
            if learned is None:
                _CACHE[key] = None
                logger.warning("TRTLLM MoE prepare cache: key %s not a row permutation; using original", key)
                return original(w13, w2, w13_scale, w2_scale, is_gated)
            perms, ref_real = learned
            _CACHE[key] = perms
            logger.info("TRTLLM MoE prepare cache: learned + validated row perms for %s", key)
            return ref_real
        if entry is None:
            return original(w13, w2, w13_scale, w2_scale, is_gated)
        return _apply(entry, (w13, w2, w13_scale, w2_scale))

    flashinfer_utils._shuffle_mxfp8_moe_weights = cached_shuffle
    _INSTALLED = True
    return True


def uninstall() -> None:
    global _INSTALLED
    if _INSTALLED and _ORIGINAL is not None:
        from vllm.model_executor.layers.quantization.utils import flashinfer_utils

        flashinfer_utils._shuffle_mxfp8_moe_weights = _ORIGINAL
        _INSTALLED = False
