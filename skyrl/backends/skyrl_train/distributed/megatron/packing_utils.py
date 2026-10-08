import math
from typing import Any, Optional

from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    AUTO_FP8_RECIPE,
    is_mxfp8_recipe,
)


def _fp8_token_align(tp_size: int, cp_size: int, fp8_recipe: Any) -> int:
    # MXFP8 quantizes sequence-parallel all-gather inputs in 1x32 tiles, so
    # every rank's local shard must hold a multiple of 32 tokens: 32*tp*cp
    # globally at any TP. Blockwise FP8 quantizes in 1x128 tiles, requiring
    # 128-token local shards under sequence parallelism (a 128*tp*cp global
    # segment when tp>1) and 16-token local slabs at TP=1.
    #
    # An unresolved "auto" must never fall through to either branch. The
    # controller's collator and the worker's preprocess_packed_seqs have to
    # derive the *same* align_size -- the collator advances
    # ``row_offset += round_up(length, align_size)`` between sub-sequences and
    # the worker re-derives those offsets to gather them back -- so any
    # disagreement silently reads the wrong tokens (no crash; see the lockstep
    # note in train/dataset/collators.py). A GPU-less driver ships "auto"
    # through unresolved while every Megatron worker resolves it locally
    # against its own device (megatron_worker.py), so guessing a branch here
    # picks a grid the workers will not share: on Blackwell they resolve to
    # mxfp8 (32-token tiles) while this would return the blockwise 16 at TP=1
    # or 128*tp*cp above it. Refuse, the way resolve_auto_wire_format does.
    if isinstance(fp8_recipe, str) and fp8_recipe.strip().lower() == AUTO_FP8_RECIPE:
        raise ValueError(
            'FP8 sequence packing cannot be aligned while fp8_recipe is still "auto" '
            "(a driver without a visible CUDA device defers recipe resolution to its "
            "workers, which would then pack on a different token grid than the "
            "controller). Set transformer_config_kwargs.fp8_recipe to 'blockwise' or "
            "'mxfp8' explicitly."
        )
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
