import math
from typing import Any


def is_fp8_enabled(fp8: Any) -> bool:
    """Return whether a Megatron/TE fp8 config value enables FP8 execution."""
    if isinstance(fp8, str):
        return fp8.strip().lower() not in {"", "0", "false", "none", "null", "no", "off"}
    return bool(fp8)


def get_packed_seq_align_size(tp_size: int, cp_size: int, fp8_enabled: bool = False) -> int:
    """Return the global alignment unit for packed TP/CP/FP8 sequences."""
    if tp_size < 1 or cp_size < 1:
        raise ValueError(f"tp_size and cp_size must be positive, got tp_size={tp_size}, cp_size={cp_size}")
    if cp_size > 1:
        layout_align = tp_size * cp_size * 2
    else:
        layout_align = tp_size
    if not fp8_enabled:
        return layout_align
    fp8_token_align = 128 * tp_size * cp_size if tp_size > 1 else 16 * cp_size
    return math.lcm(layout_align, fp8_token_align)


def get_unpacked_seq_align_size(tp_size: int, fp8_enabled: bool = False) -> int:
    """Return the alignment unit for unpacked TP/FP8 sequences without CP."""
    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if not fp8_enabled:
        return tp_size
    fp8_token_align = 128 * tp_size if tp_size > 1 else 16
    return math.lcm(tp_size, fp8_token_align)
