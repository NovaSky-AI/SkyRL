"""Selecting which modules a LoRA adapter is applied to.

Kept separate from ``workers/model_wrapper.py`` so the logic can be imported --
and unit tested -- without pulling in ``flash_attn``, which is GPU-only.
"""

from typing import Optional, Union

import torch.nn as nn
from loguru import logger

from skyrl.utils.tok import check_is_vlm

# Parent module names that identify which part of a transformer block a linear
# layer belongs to, in Llama/Qwen-style naming. The JAX backend's `filter_lora`
# (skyrl/tx/utils/models.py) uses the same names, except `linear_attn`.
LORA_ATTN_PARENT_MODULES = frozenset({"self_attn", "linear_attn"})
# `experts` matches only architectures exposing expert projections as nn.Linear.
# Current transformers fuses routed experts into 3-D nn.Parameter tensors that PEFT
# cannot target, so narrowing to the MLP group covers only shared experts and dense
# MLP layers, and raises on a model that has neither.
LORA_MLP_PARENT_MODULES = frozenset({"mlp", "experts"})
LORA_UNEMBED_MODULES = frozenset({"lm_head", "embed_tokens"})

# What `target_modules="all-linear"` expands to on the Megatron backend: Megatron-Core
# module names, in order, each with its LoRA group. Megatron-Bridge adapts every module
# whose own name is listed and only warns about a name that matches nothing, so one list
# serves every architecture. Gated DeltaNet (under `self_attention`) and Mamba (under
# `mixer`) both name their projections `in_proj`/`out_proj`; both count as attention.
MEGATRON_ALL_LINEAR_TARGETS = {
    "lora": (
        ("linear_qkv", "attn"),
        ("linear_proj", "attn"),
        ("linear_fc1", "mlp"),
        ("linear_fc2", "mlp"),
        ("in_proj", "attn"),
        ("out_proj", "attn"),
    ),
    "canonical_lora": (
        ("linear_q", "attn"),
        ("linear_k", "attn"),
        ("linear_v", "attn"),
        ("linear_proj", "attn"),
        ("linear_fc1_up", "mlp"),
        ("linear_fc1_gate", "mlp"),
        ("linear_fc2", "mlp"),
        ("in_proj", "attn"),
        ("out_proj", "attn"),
    ),
}


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


def check_lora_group_flags(
    target_modules: Union[str, list[str], None],
    train_attn: bool,
    train_mlp: bool,
    init_adapter_path: Optional[str] = None,
) -> None:
    """Reject ``train_attn``/``train_mlp`` combinations that cannot be applied.

    Needs no model, so callers can run it before reserving GPUs.

    Args:
        target_modules: The configured ``trainer.policy.model.lora.target_modules``.
        train_attn: Whether to adapt the attention group.
        train_mlp: Whether to adapt the MLP group.
        init_adapter_path: The configured ``trainer.policy.model.lora.init_adapter_path``, if any.

    Raises:
        ValueError: If both flags are ``False``, or if narrowing is requested alongside an explicit
            module list or an adapter to initialize from.
    """
    if train_attn and train_mlp:
        return
    if not train_attn and not train_mlp:
        raise ValueError(
            "LoRA requires at least one of `train_attn` or `train_mlp` to be True; with both False "
            "no module would be adapted and there would be nothing to train."
        )
    if init_adapter_path:
        raise ValueError(
            "`train_attn`/`train_mlp` narrowing cannot be combined with "
            "`trainer.policy.model.lora.init_adapter_path`: the adapter decides which modules are adapted. "
            "Set both to True."
        )
    if not (isinstance(target_modules, str) and target_modules.lower() == "all-linear"):
        raise ValueError(
            '`train_attn`/`train_mlp` narrow `trainer.policy.model.lora.target_modules="all-linear"`, but '
            f"it is set to {target_modules!r}. Name the modules in "
            '`trainer.policy.model.lora.target_modules` directly, or leave it at "all-linear" and use the '
            "flags."
        )


def check_megatron_lora_group_support(
    train_attn: bool,
    train_mlp: bool,
    *,
    has_vision_tower: bool,
    multi_latent_attention: bool,
) -> None:
    """Reject narrowing on Megatron models where the ``"all-linear"`` halves do not match the groups.

    Args:
        train_attn: Whether to adapt the attention group.
        train_mlp: Whether to adapt the MLP group.
        has_vision_tower: Whether the model being built includes a vision encoder.
        multi_latent_attention: Whether the model uses multi-latent attention (MLA).

    Raises:
        ValueError: If narrowing is requested on a model with a vision tower, or attention-only
            LoRA on an MLA model.
    """
    if train_attn and train_mlp:
        return
    # The vision tower and its patch merger reuse the language model's module names
    # (`linear_qkv`, `linear_fc1`, ...), so either half would adapt parts of it too.
    if has_vision_tower:
        raise ValueError(
            "`train_attn`/`train_mlp` narrowing is not supported for vision-language models on the "
            "Megatron backend. Set both to True and name the modules to adapt in "
            "`trainer.policy.model.lora.target_modules` instead."
        )
    if multi_latent_attention and not train_mlp:
        raise ValueError(
            "Attention-only LoRA (`train_mlp=False`) is not supported for multi-latent attention models "
            'on the Megatron backend: `trainer.policy.model.lora.target_modules="all-linear"` adapts only '
            "`linear_proj` of their attention. Set both to True and name the modules to adapt in "
            "`trainer.policy.model.lora.target_modules` instead, e.g. `linear_q_down_proj`, "
            "`linear_q_up_proj`, `linear_kv_down_proj`, `linear_kv_up_proj`, `linear_proj`."
        )


def megatron_all_linear_target_modules(
    lora_type: str,
    train_attn: bool,
    train_mlp: bool,
    include_gdn_in_proj: bool,
) -> list[str]:
    """Expand ``target_modules="all-linear"`` to Megatron-Core module names.

    Args:
        lora_type: ``"lora"`` (fused ``linear_qkv``/``linear_fc1``) or ``"canonical_lora"``
            (separate q/k/v and up/gate adapters).
        train_attn: Whether to keep the attention names.
        train_mlp: Whether to keep the MLP names.
        include_gdn_in_proj: Whether ``in_proj`` may be adapted; ``False`` for models whose Gated
            DeltaNet ``in_proj`` cannot round-trip through weight sync (see
            ``gdn_in_proj_lora_is_safe``).

    Returns:
        The module names, in a fixed order. With both flags set, this is the full list.

    Raises:
        ValueError: If ``lora_type`` is not one of the two supported types.
    """
    if lora_type not in MEGATRON_ALL_LINEAR_TARGETS:
        raise ValueError(
            f"Unknown Megatron lora_type {lora_type!r}; expected one of {sorted(MEGATRON_ALL_LINEAR_TARGETS)}."
        )
    groups = {group for group, wanted in (("attn", train_attn), ("mlp", train_mlp)) if wanted}
    return [
        name
        for name, group in MEGATRON_ALL_LINEAR_TARGETS[lora_type]
        if group in groups and (include_gdn_in_proj or name != "in_proj")
    ]


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
        target_modules: The configured value. Must be ``"all-linear"`` when narrowing to one
            group with ``train_attn`` or ``train_mlp``, since intersecting an explicit list is
            ambiguous.
        train_attn: Whether to adapt attention projections.
        train_mlp: Whether to adapt MLP projections (and expert projections, where the
            architecture exposes them as ``nn.Linear``).

    Returns:
        ``target_modules`` unchanged, or the sorted leaf names of the selected group.

    Raises:
        ValueError: If both flags are ``False``, if narrowing is requested alongside an explicit
            module list or on a vision-language model, if the model has no linear layers in the
            requested group, or if a selected name also matches a module outside it.
    """
    if train_attn and train_mlp:
        return target_modules
    check_lora_group_flags(target_modules, train_attn, train_mlp)
    # Vision towers and mergers reuse the language model's leaf names and use other
    # parent names, so the narrowed list would adapt modules outside the group.
    if check_is_vlm(getattr(model, "config", None)):
        raise ValueError(
            f"`train_attn`/`train_mlp` narrowing is not supported for vision-language models "
            f"({type(model).__name__}). Set both to True and name the modules to adapt in "
            f"`trainer.policy.model.lora.target_modules` instead."
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
            f"paths containing one of {sorted(expected)}, which this architecture does not use. "
            f"Set `train_attn` and `train_mlp` to True and name the modules to adapt in "
            f"`trainer.policy.model.lora.target_modules` instead."
        )
    # PEFT matches each name against the end of every module path, so a name the group
    # shares with a module outside it would adapt that module too.
    leaked = sorted(
        name for name, _ in model.named_modules() if name.split(".")[-1] in names and lora_module_group(name) != wanted
    )
    if leaked:
        raise ValueError(
            f"Narrowing LoRA to the {wanted} group on {type(model).__name__} would also adapt {leaked[:3]}"
            f"{' and others' if len(leaked) > 3 else ''}, which share a name with {wanted} layers. Set "
            f"`train_attn` and `train_mlp` to True and name the modules to adapt in "
            f"`trainer.policy.model.lora.target_modules` instead."
        )
    logger.info(f"LoRA restricted to {wanted} modules: {names}")
    return names


def resolve_critic_lora_target_modules(
    model: nn.Module,
    target_modules: Union[str, list[str], None],
    train_attn: bool,
    train_mlp: bool,
    value_head_name: str,
) -> Union[str, list[str], None]:
    """Like :func:`resolve_lora_target_modules`, but a narrowed critic also adapts its value head.

    The value head is in neither group. ``"all-linear"`` adapts it, and without an adapter it
    would stay frozen at its initial weights.

    Args:
        model: The loaded critic, inspected to discover its linear layers.
        target_modules: The configured value. Must be ``"all-linear"`` when narrowing to one
            group with ``train_attn`` or ``train_mlp``.
        train_attn: Whether to adapt attention projections.
        train_mlp: Whether to adapt MLP projections.
        value_head_name: Attribute name of the critic's value head (``value_head_prefix`` in
            ``get_llm_for_sequence_regression``). Appended to the targets only when exactly one
            of ``train_attn`` / ``train_mlp`` is set; with both set ``"all-linear"`` already
            covers it.

    Returns:
        ``target_modules`` unchanged when both flags are set, otherwise the selected group's
        leaf names followed by ``value_head_name``.

    Raises:
        ValueError: On the same conditions as :func:`resolve_lora_target_modules`.
    """
    targets = resolve_lora_target_modules(model, target_modules, train_attn, train_mlp)
    if train_attn and train_mlp:
        return targets
    return [*targets, value_head_name]
