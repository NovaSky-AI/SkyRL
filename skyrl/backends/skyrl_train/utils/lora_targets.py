"""Selecting which modules a LoRA adapter is applied to.

Kept separate from ``workers/model_wrapper.py`` so the logic can be imported --
and unit tested -- without pulling in ``flash_attn``, which is GPU-only.
"""

from typing import Optional, Union

import torch.nn as nn
from loguru import logger

# Parent module names that identify which part of a transformer block a linear
# layer belongs to. Mirrors the JAX backend's `filter_lora`
# (skyrl/tx/utils/models.py), so both backends agree on what "attention" and
# "MLP" mean for Tinker's train_attn / train_mlp flags.
LORA_ATTN_PARENT_MODULES = frozenset({"self_attn", "linear_attn"})
# `experts` matches only architectures exposing expert projections as nn.Linear;
# current transformers fuses them into 3-D nn.Parameter tensors that PEFT cannot
# target, so on those models the MLP group is empty and narrowing to it raises.
LORA_MLP_PARENT_MODULES = frozenset({"mlp", "experts"})
LORA_UNEMBED_MODULES = frozenset({"lm_head", "embed_tokens"})


def lora_module_group(module_name: str) -> Optional[str]:
    """Classify a dotted module path into the LoRA group it belongs to.

    Args:
        module_name: Dotted module path, e.g. ``model.layers.0.self_attn.q_proj``.

    Returns:
        ``"attn"``, ``"mlp"``, ``"unembed"``, or ``None`` when the module belongs to none of them.
    """
    parts = set(module_name.split("."))
    if parts & LORA_UNEMBED_MODULES:
        return "unembed"
    if parts & LORA_ATTN_PARENT_MODULES:
        return "attn"
    if parts & LORA_MLP_PARENT_MODULES:
        return "mlp"
    return None


def resolve_lora_target_modules(
    model: nn.Module,
    target_modules: Union[str, list[str], None],
    train_attn: bool,
    train_mlp: bool,
) -> Union[str, list[str], None]:
    """Narrow ``target_modules`` to the attention or MLP layers of ``model``.

    With both flags set (the default) ``target_modules`` is returned untouched, so the common path
    is identical to the behaviour before train_attn / train_mlp existed. PEFT's ``"all-linear"``
    already skips the output layer, so the result never covers ``lm_head``.

    Args:
        model: The loaded model, inspected to discover its linear layers.
        target_modules: The configured value. Must be ``"all-linear"`` when narrowing, since
            intersecting an explicit list is ambiguous.
        train_attn: Whether to adapt attention projections.
        train_mlp: Whether to adapt MLP projections (and expert projections, where the
            architecture exposes them as ``nn.Linear``).

    Returns:
        ``target_modules`` unchanged, or the sorted leaf names of the selected group.

    Raises:
        ValueError: If both flags are ``False``, if narrowing is requested alongside an explicit
            module list, or if the model has no linear layers in the requested group.
    """
    if train_attn and train_mlp:
        return target_modules
    if not train_attn and not train_mlp:
        raise ValueError(
            "LoRA requires at least one of `train_attn` or `train_mlp` to be True; with both False "
            "no module would be adapted and there would be nothing to train."
        )
    if target_modules != "all-linear":
        raise ValueError(
            f"`train_attn`/`train_mlp` narrow `target_modules='all-linear'`, but it is set to "
            f"{target_modules!r}. Name the modules in `target_modules` directly, or leave it at "
            f'"all-linear" and use the flags.'
        )

    wanted = "attn" if train_attn else "mlp"
    names = sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and lora_module_group(name) == wanted
        }
    )
    if not names:
        expected = LORA_ATTN_PARENT_MODULES if wanted == "attn" else LORA_MLP_PARENT_MODULES
        raise ValueError(
            f"No {wanted} linear layers found for LoRA on {type(model).__name__}. Expected module "
            f"paths containing one of {sorted(expected)}."
        )
    logger.info(f"LoRA restricted to {wanted} modules: {names}")
    return names
