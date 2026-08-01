import math
from typing import Any, MutableMapping, Optional

AUTO_FP8_RECIPE = "auto"


def is_fp8_enabled(fp8: Any) -> bool:
    """Return whether a Megatron/TE fp8 config value enables FP8 execution."""
    if isinstance(fp8, str):
        return fp8.strip().lower() not in {"", "0", "false", "none", "null", "no", "off"}
    return bool(fp8)


def is_mxfp8_recipe(fp8_recipe: Any) -> bool:
    """Return whether a Megatron/TE fp8 recipe value selects MXFP8."""
    return isinstance(fp8_recipe, str) and fp8_recipe.strip().lower() == "mxfp8"


def is_blackwell_or_newer() -> bool:
    """Return whether the visible CUDA device is SM100+ (Blackwell or newer)."""
    import torch

    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 10


def resolve_auto_fp8_recipe(transformer_config_kwargs: Optional[MutableMapping[str, Any]]) -> Any:
    """Resolve ``fp8_recipe="auto"`` to the architecture-native TE recipe.

    Mutates ``transformer_config_kwargs`` in place and returns the resolved
    recipe; any explicitly configured recipe is returned unchanged. ``"auto"``
    selects the recipe each architecture supports natively: ``blockwise``
    (``Float8BlockScaling``, 1x128/128x128 tiles with FP32 scales) on Hopper,
    and ``mxfp8`` (``MXFP8BlockScaling``, hardware 1x32 tiles with E8M0
    scales) on Blackwell — where TE can only emulate the blockwise recipe on
    the MX datapath with power-of-2 scales, the configuration that needs the
    ``NVTE_FP8_BLOCK_AMAX_EPSILON`` floor for zero-token MoE experts. Without
    a visible CUDA device ``"auto"`` falls back to ``blockwise``, which runs
    on both architectures.
    """
    recipe = transformer_config_kwargs.get("fp8_recipe") if transformer_config_kwargs else None
    if not isinstance(recipe, str) or recipe.strip().lower() != AUTO_FP8_RECIPE:
        return recipe
    resolved = "mxfp8" if is_blackwell_or_newer() else "blockwise"
    transformer_config_kwargs["fp8_recipe"] = resolved
    return resolved


def _fp8_token_align(tp_size: int, cp_size: int, fp8_recipe: Any) -> int:
    # Blockwise FP8 quantizes sequence-parallel all-gather inputs in 1x128
    # tiles (a 128*tp*cp global segment when tp>1) and requires 16-token
    # local slabs at TP=1. MXFP8 quantizes in 1x32 tiles, so the TP=1 slab
    # grows to 32 (TE asserts dims % 32 == 0); the tp>1 segment of
    # 128*tp*cp is already a multiple of 32 and stays unchanged.
    if tp_size > 1:
        return 128 * tp_size * cp_size
    return (32 if is_mxfp8_recipe(fp8_recipe) else 16) * cp_size


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
