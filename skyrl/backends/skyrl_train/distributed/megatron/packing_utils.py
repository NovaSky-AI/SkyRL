import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    is_mxfp8_recipe,
)


def _fp8_token_align(tp_size: int, cp_size: int, fp8_recipe: Any) -> int:
    # MXFP8 quantizes sequence-parallel all-gather inputs in 1x32 tiles, so
    # every rank's local shard must hold a multiple of 32 tokens: 32*tp*cp
    # globally at any TP. Blockwise FP8 quantizes in 1x128 tiles, requiring
    # 128-token local shards under sequence parallelism (a 128*tp*cp global
    # segment when tp>1) and 16-token local slabs at TP=1.
    if is_mxfp8_recipe(fp8_recipe):
        return 32 * tp_size * cp_size
    if tp_size > 1:
        return 128 * tp_size * cp_size
    return 16 * cp_size


def get_packing_align_size_sequence(tp_size: int, cp_size: int) -> int:
    """Return the alignment required independently for each packed sequence."""
    if tp_size < 1 or cp_size < 1:
        raise ValueError(f"tp_size and cp_size must be positive, got tp_size={tp_size}, cp_size={cp_size}")
    if cp_size > 1:
        return tp_size * cp_size * 2
    return 1


def get_packing_align_size_total(
    tp_size: int, cp_size: int, fp8_enabled: bool = False, fp8_recipe: Optional[str] = None
) -> int:
    """Return the alignment required for the aggregate packed token slab."""
    if tp_size < 1 or cp_size < 1:
        raise ValueError(f"tp_size and cp_size must be positive, got tp_size={tp_size}, cp_size={cp_size}")
    if cp_size > 1:
        layout_align = tp_size * cp_size * 2
    else:
        layout_align = tp_size
    if not fp8_enabled:
        return layout_align
    return math.lcm(layout_align, _fp8_token_align(tp_size, cp_size, fp8_recipe))


def get_unpacked_seq_align_size(tp_size: int, fp8_enabled: bool = False, fp8_recipe: Optional[str] = None) -> int:
    """Return the alignment unit for unpacked TP/FP8 sequences without CP."""
    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if not fp8_enabled:
        return tp_size
    return math.lcm(tp_size, _fp8_token_align(tp_size, 1, fp8_recipe))


@dataclass(frozen=True)
class PackedSegmentLayout:
    """Where every packed sub-sequence starts and ends once alignment is applied."""

    padded_lengths: tuple[int, ...]
    cu_seqlens_padded: tuple[int, ...]

    @property
    def total(self) -> int:
        return self.cu_seqlens_padded[-1]


def packed_segment_layout(
    sequence_lengths: Sequence[int],
    *,
    tp_size: int,
    cp_size: int,
    fp8_enabled: bool = False,
    fp8_recipe: Optional[str] = None,
) -> PackedSegmentLayout:
    """Resolve the padded packed layout for one batch of sub-sequence lengths.

    The single source of truth for two decisions every packing site depends on: each
    sub-sequence is padded to the sequence alignment so flash-attn varlen sees aligned
    segment boundaries, and the aggregate slab alignment that TP and FP8 need is then
    applied **once, by growing the final sub-sequence** — not by padding every one of
    them, and not as a separate trailing segment. Pure integer arithmetic so the host
    metadata builders, the controller collator and the worker can all share it.
    """
    sequence_align = get_packing_align_size_sequence(tp_size, cp_size)
    total_align = get_packing_align_size_total(tp_size, cp_size, fp8_enabled=fp8_enabled, fp8_recipe=fp8_recipe)

    padded = []
    for raw_length in sequence_lengths:
        length = int(raw_length)
        if length <= 0:
            raise ValueError(f"Packed sub-sequence lengths must be positive, got {length}")
        padded.append(length + (-length % sequence_align))
    if not padded:
        return PackedSegmentLayout(padded_lengths=(), cu_seqlens_padded=(0,))

    padded[-1] += -sum(padded) % total_align

    offsets = [0]
    for length in padded:
        offsets.append(offsets[-1] + length)
    return PackedSegmentLayout(padded_lengths=tuple(padded), cu_seqlens_padded=tuple(offsets))
