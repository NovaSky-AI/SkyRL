"""Generic per-model spec for serialized FP8 weight sync (blockwise and MXFP8 wires).

``ModelFp8Spec`` groups everything the sync path must know about one model
family — which HF configs it matches, which weights quantize, which vLLM
modules stay unquantized, and how Megatron-Bridge's batched expert tensors
map onto wire projections. ``resolve_fp8_spec`` selects the spec for a
checkpoint; unsupported layouts resolve to ``None`` and callers reject them
explicitly. The vLLM-side fused-loader targets are derived from the same
projections via ``batched_moe_wire_targets``, so sender and receiver share
one source of truth instead of hardcoding the mapping twice.

User exclusions (``fp8_weight_sync_exclude_modules``) are layered on top of
the spec's ``base_exclude_list``: ``resolve_user_provided_exclude_list``
expands them into module names. The driver (for the engine,
``engine_exclude_list``) and every trainer rank (for the sender,
``SerializedFp8Config.user_provided_exclude_list``) call it on the same
config, so both sides see the same modules.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class MoeProjection:
    """One routed-expert projection and its vLLM fused-loader target."""

    hf_name: str  # projection name on the wire / in HF checkpoints, e.g. "gate_proj"
    vllm_param: str  # fused vLLM parameter it loads into, e.g. "w13_weight"
    shard_id: str  # FusedMoE weight_loader shard id, e.g. "w1"


@dataclass(frozen=True)
class MoeExpertSpec:
    """A Megatron-Bridge batched expert tensor mapped onto wire projections.

    ``split_dim`` names the tensor dimension that concatenates the
    projections (split evenly, in order); ``None`` means the tensor is a
    single projection.
    """

    experts_base: str  # checkpoint prefix ending in the experts module
    projections: tuple[MoeProjection, ...]
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
    # hf_config -> HF module names synced as FP8, as Megatron-Bridge exports them;
    # a layer's routed experts are one entry whose last segment is moe_module.
    # User exclusion globs match against these names.
    fp8_modules: Callable[[Any], list[str]]
    # batched expert tensor name -> MoeExpertSpec, or None if not one
    moe_expert_spec: Callable[[str], Optional[MoeExpertSpec]]
    # module segment holding routed experts in vLLM parameter names
    moe_module: str = "experts"
    # every projection the model emits, for receiver-side target derivation
    moe_projections: tuple[MoeProjection, ...] = field(default=())


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


def resolve_user_provided_exclude_list(spec: ModelFp8Spec, hf_config: Any, patterns: Sequence[str]) -> tuple[str, ...]:
    """Expand the user's exclusion globs into the HF module names they keep unquantized.

    Args:
        spec (ModelFp8Spec): The checkpoint's model spec.
        hf_config (Any): The checkpoint's HF config.
        patterns (Sequence[str]): HF module-name globs such as ``["*.layers.3.mlp.*"]``.

    Returns:
        tuple[str, ...]: The matched module names in model order, without duplicates. On a
            Qwen3.5 MoE checkpoint, ``["*.layers.3.mlp.*"]`` gives
            ``(
                "model.language_model.layers.3.mlp.experts",
                "model.language_model.layers.3.mlp.shared_expert.gate_proj",
                ...
            )``
    """

    modules = spec.fp8_modules(hf_config)
    excluded: set[str] = set()
    for pattern in patterns:
        matched = [module for module in modules if fnmatch.fnmatchcase(module, pattern)]
        excluded.update(matched)
    return tuple(module for module in modules if module in excluded)


def engine_exclude_list(
    spec: ModelFp8Spec,
    hf_config: Any,
    wire_format: str,
    user_provided_exclude_list: Sequence[str],
) -> list[str]:
    """Return the modules the engine builds unquantized: the spec's base list plus the user's.

    vLLM picks one scheme for all of a layer's routed experts. Its compressed-tensors
    config (MXFP8) decides from expert 0's projection names, matched exactly; its fp8
    config (blockwise) matches any ignored name containing the experts' prefix. An
    excluded experts module is therefore listed as expert 0's projections.
    """

    exclude_list = list(spec.base_exclude_list(hf_config, wire_format))
    for module in user_provided_exclude_list:
        if module.rpartition(".")[2] == spec.moe_module:
            exclude_list.extend(f"{module}.0.{proj.hf_name}" for proj in spec.moe_projections)
        else:
            exclude_list.append(module)
    return exclude_list


def batched_moe_wire_targets() -> dict[str, tuple[str, str]]:
    """Receiver mapping: checkpoint suffix -> (fused vLLM suffix, shard id).

    Derived from every registered spec's projections so the vLLM worker
    extension never re-encodes per-model fused-loader knowledge.
    """

    targets: dict[str, tuple[str, str]] = {}
    for spec in _REGISTRY:
        for proj in spec.moe_projections:
            # Blockwise ships weight_scale_inv; MXFP8 ships compressed-tensors'
            # weight_scale. Both map onto the same fused parameter family.
            for weight_suffix, param_suffix in (
                (".weight", ""),
                (WIRE_SCALE_SUFFIX[BLOCKWISE_FP8], "_scale_inv"),
                (WIRE_SCALE_SUFFIX[MXFP8], "_scale"),
            ):
                key = f".{spec.moe_module}.{proj.hf_name}{weight_suffix}"
                value = (f".{spec.moe_module}.{proj.vllm_param}{param_suffix}", proj.shard_id)
                if targets.setdefault(key, value) != value:
                    raise ValueError(f"Conflicting batched MoE wire target registered for suffix {key!r}")
    return targets
