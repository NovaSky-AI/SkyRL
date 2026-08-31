import math
from typing import Any


def is_fp8_enabled(fp8: Any) -> bool:
    """Return whether a Megatron/TE fp8 config value enables FP8 execution."""
    if isinstance(fp8, str):
        return fp8.strip().lower() not in {"", "0", "false", "none", "null", "no"}
    return bool(fp8)


def get_packing_align_size_sequence(tp_size: int, cp_size: int) -> int:
    """Return the alignment required independently for each packed sequence."""
    if cp_size > 1:
        return tp_size * cp_size * 2
    return 1


def get_packing_align_size_total(tp_size: int, cp_size: int, fp8_enabled: bool = False) -> int:
    """Return the alignment required for the aggregate packed token slab."""
    if cp_size > 1:
        layout_align = tp_size * cp_size * 2
    else:
        layout_align = tp_size
    if not fp8_enabled:
        return layout_align
    return math.lcm(layout_align, 16 * cp_size)


def get_unpacked_seq_align_size(tp_size: int, fp8_enabled: bool = False) -> int:
    """Return sequence padding needed when removing microbatch padding without CP."""
    if not fp8_enabled:
        return tp_size
    return math.lcm(tp_size, 16)
