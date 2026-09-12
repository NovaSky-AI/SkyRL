import math
from typing import Any, Optional

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


def get_packed_seq_align_size(
    tp_size: int, cp_size: int, fp8_enabled: bool = False, fp8_recipe: Optional[str] = None
) -> int:
    """Return the global alignment unit for packed TP/CP/FP8 sequences."""
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
