"""Generic per-model spec for serialized blockwise FP8 weight sync.

``ModelFp8Spec`` groups everything the sync path must know about one model
family — which HF configs it matches, which weights quantize, which vLLM
modules stay unquantized, and how Megatron-Bridge's batched expert tensors
map onto wire projections. ``resolve_fp8_spec`` selects the spec for a
checkpoint; unsupported layouts resolve to ``None`` and callers reject them
explicitly. The vLLM-side fused-loader targets are derived from the same
projections via ``batched_moe_wire_targets``, so sender and receiver share
one source of truth instead of hardcoding the mapping twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence


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
    """Per-model policy for serialized blockwise FP8 weight sync."""

    name: str
    # hf_config -> does this spec support the checkpoint layout?
    matches: Callable[[Any], bool]
    # (hf_name, shape) -> serialize this exported weight as FP8?
    should_quantize: Callable[[str, Sequence[int]], bool]
    # hf_config -> vLLM module prefixes that must stay unquantized
    ignored_layers: Callable[[Any], list[str]]
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


def batched_moe_wire_targets() -> dict[str, tuple[str, str]]:
    """Receiver mapping: checkpoint suffix -> (fused vLLM suffix, shard id).

    Derived from every registered spec's projections so the vLLM worker
    extension never re-encodes per-model fused-loader knowledge.
    """

    targets: dict[str, tuple[str, str]] = {}
    for spec in _REGISTRY:
        for proj in spec.moe_projections:
            for weight_suffix, param_suffix in ((".weight", ""), (".weight_scale_inv", "_scale_inv")):
                key = f".{spec.moe_module}.{proj.hf_name}{weight_suffix}"
                value = (f".{spec.moe_module}.{proj.vllm_param}{param_suffix}", proj.shard_id)
                if targets.setdefault(key, value) != value:
                    raise ValueError(f"Conflicting batched MoE wire target registered for suffix {key!r}")
    return targets
