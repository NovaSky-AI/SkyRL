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


def validate_mxfp8_gdn_tp_alignment(
    transformer_config_kwargs: Optional[MutableMapping[str, Any]],
    hf_config: Any,
    tensor_model_parallel_size: int,
) -> None:
    """Refuse mxfp8 + GDN + TP combinations whose weight shard TE cannot quantize.

    Megatron fuses the GDN q/k/v/z/b/a projections into ONE column-parallel
    ``in_proj`` GEMM (megatron/core/ssm/gated_delta_net.py)::

        in_proj_dim = 2*qk_dim + 2*v_dim + 2*num_value_heads

    TP shards that output dim, so each rank quantizes an
    ``[in_proj_dim // TP, hidden]`` weight, and MXFP8 scales cover 1x32 groups,
    so TE requires every dim of the *shard* to be a multiple of 32
    (``MXFP8Quantizer::create_tensor``, transformer_engine/pytorch/csrc/quantizer.cpp).
    Megatron's own guard checks only the GLOBAL dim, which can pass while the
    shard is misaligned — Qwen3.5's 12352 = 32*386 shards to 6176 at TP=2 (fine)
    but 3088 at TP=4 (rejected) — and the failure then surfaces as a TE C++
    assert deep inside model build. Catch it here, with the arithmetic.
    """
    kwargs = transformer_config_kwargs or {}
    if not is_fp8_enabled(kwargs.get("fp8")) or not is_mxfp8_recipe(kwargs.get("fp8_recipe")):
        return
    config = getattr(hf_config, "text_config", None) or hf_config
    dims = [
        getattr(config, name, None)
        for name in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
        )
    ]
    if any(dim is None for dim in dims):
        return  # no GDN linear-attention layers in this model
    num_key_heads, num_value_heads, key_head_dim, value_head_dim = dims
    in_proj_dim = 2 * key_head_dim * num_key_heads + 2 * value_head_dim * num_value_heads + 2 * num_value_heads
    tp = tensor_model_parallel_size
    if in_proj_dim % (32 * tp) == 0:
        return
    raise ValueError(
        f"fp8_recipe='mxfp8' cannot run this GDN model at tensor_model_parallel_size={tp}: "
        f"Megatron fuses the GDN q/k/v/z/b/a projections into one column-parallel in_proj GEMM "
        f"(megatron/core/ssm/gated_delta_net.py) with output dim {in_proj_dim} = "
        f"2*{key_head_dim}*{num_key_heads} + 2*{value_head_dim}*{num_value_heads} + 2*{num_value_heads}, "
        f"each TP rank quantizes an [{in_proj_dim}/{tp}, hidden] shard, and TE's MXFP8 quantizer "
        f"(MXFP8Quantizer::create_tensor) rejects any dim not divisible by 32. Pick a "
        f"tensor_model_parallel_size that divides in_proj_dim/32 = {in_proj_dim / 32:g}, "
        f"or use fp8_recipe='blockwise'."
    )


def resolve_auto_wire_format(fp8_recipe: Any) -> str:
    """Resolve ``fp8_weight_sync_mode="auto"`` from the policy's concrete recipe.

    Keys off the *recipe*, never the architecture: the rollout must serve the
    representation the trainer computes with, so an explicit blockwise recipe
    on Blackwell keeps a blockwise wire. Call after ``resolve_auto_fp8_recipe``;
    a recipe left ``"auto"`` by a GPU-less driver is refused rather than
    guessed, because the wire chosen here reaches every engine's boot config —
    unlike the recipe, it gets no second resolution on the workers.
    """
    if isinstance(fp8_recipe, str) and fp8_recipe.strip().lower() == AUTO_FP8_RECIPE:
        raise ValueError(
            'fp8_weight_sync_mode="auto" cannot be resolved while fp8_recipe is still "auto" '
            "(a driver without a visible CUDA device defers recipe resolution to its workers). "
            "Set transformer_config_kwargs.fp8_recipe to 'blockwise' or 'mxfp8', or set "
            "fp8_weight_sync_mode explicitly."
        )
    # Lazy import: the wire-format constants live in weight_sync, whose package
    # init pulls torch; this module stays importable without it.
    from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
        BLOCKWISE_FP8,
        MXFP8,
    )

    return MXFP8 if is_mxfp8_recipe(fp8_recipe) else BLOCKWISE_FP8


def wire_to_engine_quantization(wire_format: str) -> str:
    """Map a serialized wire format to the vLLM ``quantization`` method that serves it.

    MXFP8 rides vLLM's compressed-tensors path; blockwise rides its fp8 path.
    Engines need this value at boot (``engine_init_kwargs.quantization``),
    before trainer-side config resolution runs, so launchers deriving it for
    ``fp8_weight_sync_mode=auto`` call this instead of re-encoding the mapping.
    """
    from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
        MXFP8,
        WIRE_FORMATS,
    )

    if wire_format not in WIRE_FORMATS:
        raise ValueError(f"Unsupported wire format {wire_format!r}; expected one of {WIRE_FORMATS!r}")
    return "compressed-tensors" if wire_format == MXFP8 else "fp8"
