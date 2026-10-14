"""Generic per-model spec for serialized FP8 weight sync (blockwise and MXFP8 wires).

``ModelFp8Spec`` groups everything the sync path must know about one model
family — which HF configs it matches, which weights quantize, which vLLM
modules stay unquantized, and which exported tensors stack a layer's routed
experts. ``resolve_fp8_spec`` selects the spec for a checkpoint; unsupported
layouts resolve to ``None`` and callers reject them explicitly. Where vLLM
loads each projection of a stacked expert tensor is fixed by vLLM's FusedMoE,
not by the model, so the receiver's table (``batched_moe_wire_targets``) is
built from one constant here.

User exclusions (``fp8_weight_sync_exclude_modules``) are layered on top of
the spec's ``base_exclude_list``: ``resolve_user_provided_exclude_list``
expands them into names from the spec's ``fp8_modules``. The driver (for the
engine, ``engine_exclude_list``) and every trainer rank (for the sender,
``SerializedFp8Config.user_provided_exclude_list``) call it on the same
config, so both sides see the same tensors.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

# Wire formats live here rather than in vllm_format so model specs can name them
# without importing the serializer that imports this module.
BLOCKWISE_FP8 = "blockwise"
MXFP8 = "mxfp8"
AUTO_FP8 = "auto"
WIRE_FORMATS = (BLOCKWISE_FP8, MXFP8)

# Scale tensor suffix each wire pairs with a quantized ``.weight`` — the one
# mapping both the serializer (name emission) and the batched-MoE receiver
# (target routing) consume.
WIRE_SCALE_SUFFIX = {
    BLOCKWISE_FP8: ".weight_scale_inv",
    MXFP8: ".weight_scale",
}


# vLLM's FusedMoE slot for each projection of a stacked expert tensor on the batched
# wire: projection -> (fused parameter, shard id). Fixed by vLLM
# (``build_expert_params_mapping``), the same for every model.
_FUSED_MOE_TARGETS = {
    "gate_proj": ("w13_weight", "w1"),
    "up_proj": ("w13_weight", "w3"),
    "down_proj": ("w2_weight", "w2"),
}

# vLLM builds all routed experts of a layer as one module, at one precision. Listing
# expert 0's three projections makes both wires build that module unquantized; for
# model.layers.3.mlp.experts that is model.layers.3.mlp.experts.0.{gate,up,down}_proj:
#   - mxfp8 (compressed-tensors) checks exactly these three names and leaves the layer
#     unquantized when all are listed
#   - blockwise (fp8) leaves it unquantized when any listed name contains the experts
#     module's name, which these three do.
_VLLM_EXPERT_PRECISION_NAMES = ("0.gate_proj", "0.up_proj", "0.down_proj")


@dataclass(frozen=True)
class StackedExperts:
    """An exported tensor that stacks one projection, or two concatenated ones, of every
    routed expert in a layer, as Megatron-Bridge exports Qwen3.5's experts.

    ``split_dim`` names the tensor dimension that concatenates the
    projections (split evenly, in order); ``None`` means the tensor is a
    single projection.
    """

    experts_base: str  # checkpoint prefix ending in the experts module
    projections: tuple[str, ...]  # in stacking order, e.g. ("gate_proj", "up_proj")
    split_dim: Optional[int] = None


@dataclass(frozen=True)
class ModelFp8Spec:
    """Per-model policy for serialized FP8 weight sync; wire-format-aware
    callbacks receive the concrete format so one spec serves both wires."""

    name: str
    # hf_config -> does this spec support the checkpoint layout?
    matches: Callable[[Any], bool]
    # (hf_name, shape, wire_format) -> serialize this exported weight as FP8?
    should_quantize: Callable[[str, Sequence[int], str], bool]
    # (hf_config, wire_format) -> modules the engine always builds unquantized, before
    # any user exclusions; vLLM matches these names as module prefixes
    base_exclude_list: Callable[[Any, str], list[str]]
    # (hf_config, wire_format) -> one name per tensor the sender sends as FP8, as
    # Megatron-Bridge exports it minus a trailing ".weight". User exclusion globs match
    # against these.
    fp8_modules: Callable[[Any, str], list[str]]
    # exported tensor name -> StackedExperts, or None if it does not stack experts
    stacked_experts: Callable[[str], Optional[StackedExperts]] = lambda name: None


_REGISTRY: list[ModelFp8Spec] = []


def register_fp8_spec(spec: ModelFp8Spec) -> ModelFp8Spec:
    """Register a model spec for ``resolve_fp8_spec`` lookup."""

    if any(existing.name == spec.name for existing in _REGISTRY):
        raise ValueError(f"An FP8 model spec named {spec.name!r} is already registered")
    _REGISTRY.append(spec)
    return spec


def registered_fp8_spec_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in _REGISTRY)


def resolve_fp8_spec(hf_config: Any) -> Optional[ModelFp8Spec]:
    """Return the registered spec matching an HF config, or ``None``."""

    for spec in _REGISTRY:
        if spec.matches(hf_config):
            return spec
    return None


def _routed_experts_module(name: str) -> Optional[str]:
    """Return the experts module a routed-expert tensor belongs to, or None for any other.

    ``model.layers.3.mlp.experts.17.gate_proj`` and ``model.layers.3.mlp.experts.gate_up_proj``
    both give ``model.layers.3.mlp.experts``; ``model.layers.3.mlp.shared_experts.gate_proj``
    gives None. "experts" is vLLM's name for every FusedMoE module.
    """

    head, found, _ = name.partition(".experts.")
    return f"{head}.experts" if found else None


def resolve_user_provided_exclude_list(
    spec: ModelFp8Spec,
    hf_config: Any,
    wire_format: str,
    patterns: Sequence[str],
) -> tuple[str, ...]:
    """Expand the user's exclusion globs into the FP8 tensors they keep unquantized.

    Args:
        spec (ModelFp8Spec): The checkpoint's model spec.
        hf_config (Any): The checkpoint's HF config.
        wire_format (str): The serialized FP8 wire, ``"blockwise"`` or ``"mxfp8"``.
        patterns (Sequence[str]): Globs over the spec's ``fp8_modules`` names, such as
            ``["*.layers.3.mlp.*"]``.

    Raises:
        ValueError: A pattern matches none of the tensors the spec syncs as FP8, or the
            patterns exclude only some of a layer's routed experts. vLLM runs all of a
            layer's routed experts at one precision

    Returns:
        tuple[str, ...]: The matched names, without duplicates. On a Qwen3.5 MoE
            checkpoint, ``["*.layers.3.mlp.*"]`` gives
            ``(
                "model.language_model.layers.3.mlp.experts.gate_up_proj",
                "model.language_model.layers.3.mlp.experts.down_proj",
                "model.language_model.layers.3.mlp.shared_expert.gate_proj",
                ...
            )``
    """

    modules = spec.fp8_modules(hf_config, wire_format)
    excluded: set[str] = set()
    for pattern in patterns:
        matched = [module for module in modules if fnmatch.fnmatchcase(module, pattern)]
        if not matched:
            example = f" (e.g. {modules[0]!r})" if modules else ""
            raise ValueError(
                f"fp8_weight_sync_exclude_modules pattern {pattern!r} matches none of the "
                f"{len(modules)} tensors {spec.name} syncs as FP8{example}."
            )
        excluded.update(matched)

    experts_tensors: dict[str, list[str]] = {}
    for module in modules:
        experts = _routed_experts_module(module)
        if experts is not None:
            experts_tensors.setdefault(experts, []).append(module)
    for experts, tensors in experts_tensors.items():
        kept = [tensor for tensor in tensors if tensor not in excluded]
        if 0 < len(kept) < len(tensors):
            raise ValueError(
                f"fp8_weight_sync_exclude_modules excludes {len(tensors) - len(kept)} of the {len(tensors)} "
                f"routed-expert tensors in {experts!r}, but not {kept[0]!r}. vLLM runs all of a layer's "
                f"routed experts at one precision: exclude all of them (e.g. '{experts}.*') or none."
            )
    return tuple(excluded)


def engine_exclude_list(
    spec: ModelFp8Spec,
    hf_config: Any,
    wire_format: str,
    user_provided_exclude_list: Sequence[str],
) -> list[str]:
    """Return the modules the engine builds unquantized: the spec's base list plus the user's.

    A layer's excluded routed experts (all of them, per
    ``resolve_user_provided_exclude_list``) are listed as the names vLLM reads to choose
    their precision.
    """

    exclude_list = list(spec.base_exclude_list(hf_config, wire_format))
    listed_experts: set[str] = set()
    for module in user_provided_exclude_list:
        experts = _routed_experts_module(module)
        if experts is None:
            exclude_list.append(module)
        elif experts not in listed_experts:
            listed_experts.add(experts)
            exclude_list.extend(f"{experts}.{name}" for name in _VLLM_EXPERT_PRECISION_NAMES)
    return exclude_list


def batched_moe_wire_targets() -> dict[str, tuple[str, str]]:
    """Receiver mapping: batched wire suffix -> (fused vLLM suffix, shard id)."""

    targets: dict[str, tuple[str, str]] = {}
    for projection, (param, shard_id) in _FUSED_MOE_TARGETS.items():
        # Blockwise ships weight_scale_inv; MXFP8 ships compressed-tensors'
        # weight_scale. Both map onto the same fused parameter family.
        for weight_suffix, param_suffix in (
            (".weight", ""),
            (WIRE_SCALE_SUFFIX[BLOCKWISE_FP8], "_scale_inv"),
            (WIRE_SCALE_SUFFIX[MXFP8], "_scale"),
        ):
            targets[f".experts.{projection}{weight_suffix}"] = (f".experts.{param}{param_suffix}", shard_id)
    return targets
