"""Recipe/architecture helpers shared by FP8 packing, weight sync, and workers."""

from typing import Any, MutableMapping, Optional

from loguru import logger

AUTO_FP8_RECIPE = "auto"


def is_fp8_enabled(fp8: Any) -> bool:
    """Return whether a Megatron/TE fp8 config value enables FP8 execution."""
    if isinstance(fp8, str):
        return fp8.strip().lower() not in {"", "0", "false", "none", "null", "no", "off"}
    return bool(fp8)


def is_mxfp8_recipe(fp8_recipe: Any) -> bool:
    """Return whether a Megatron/TE fp8 recipe value selects MXFP8."""
    return isinstance(fp8_recipe, str) and fp8_recipe.strip().lower() == "mxfp8"


def has_visible_cuda_device() -> bool:
    """Return whether this process can see a CUDA device."""
    import torch

    return torch.cuda.is_available()


def is_blackwell_or_newer() -> bool:
    """Return whether the visible CUDA device is SM100+ (Blackwell or newer)."""
    import torch

    if not has_visible_cuda_device():
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
    scales) on Blackwell.

    A process with no visible CUDA device cannot know the workers'
    architecture, so it leaves ``"auto"`` in place instead of guessing — a
    GPU-less Ray driver must not bake ``blockwise`` into the config that
    Blackwell workers receive. Each Megatron worker resolves again locally
    before the value reaches TE.
    """
    recipe = transformer_config_kwargs.get("fp8_recipe") if transformer_config_kwargs else None
    if not isinstance(recipe, str) or recipe.strip().lower() != AUTO_FP8_RECIPE:
        if not is_mxfp8_recipe(recipe) and isinstance(recipe, str) and recipe.strip() and is_blackwell_or_newer():
            # TE emulates the blockwise recipe on Blackwell's MX datapath with
            # power-of-2 scales; MoE experts that receive zero tokens then emit
            # amax==0 grad blocks whose degenerate scales overflow the grad
            # norm. The native recipe has no such mode.
            logger.warning(
                "fp8_recipe={} is emulated on SM100+; prefer fp8_recipe='mxfp8' (or 'auto').",
                recipe,
            )
        return recipe
    if not has_visible_cuda_device():
        logger.info('fp8_recipe="auto" left unresolved: no CUDA device visible; each worker resolves it locally.')
        return AUTO_FP8_RECIPE
    resolved = "mxfp8" if is_blackwell_or_newer() else "blockwise"
    transformer_config_kwargs["fp8_recipe"] = resolved
    return resolved


def validate_concrete_fp8_recipe(transformer_config_kwargs: Optional[MutableMapping[str, Any]]) -> None:
    """Reject FP8 recipe/device combinations Transformer Engine cannot run.

    Runs wherever the recipe becomes concrete: on the driver when it has a
    CUDA device, and on every Megatron worker after its local resolution —
    which covers GPU-less drivers that ship ``"auto"`` through unresolved.
    """
    recipe = transformer_config_kwargs.get("fp8_recipe") if transformer_config_kwargs else None
    if not is_mxfp8_recipe(recipe):
        return
    if has_visible_cuda_device() and not is_blackwell_or_newer():
        raise ValueError(
            "fp8_recipe=mxfp8 requires SM100+ (Blackwell): TE's MXFP8BlockScaling has "
            "no pre-Blackwell kernel path. Use fp8_recipe='blockwise' (or 'auto') on Hopper."
        )
    if is_fp8_enabled(transformer_config_kwargs.get("fp8_param")):
        raise ValueError(
            "fp8_param=true is not supported with fp8_recipe=mxfp8: Transformer Engine "
            "2.11 cannot re-point MXFP8Tensor storage (replace_raw_data raises "
            "NotImplementedError), and Megatron's reuse_grad_buf_for_mxfp8_param_ag "
            "fallback zeroes freshly loaded persistent params on the first param "
            "all-gather. Use fp8_param=false with mxfp8."
        )
