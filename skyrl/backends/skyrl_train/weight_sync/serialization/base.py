"""Shared abstractions for serialized weight quantization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

LEGACY_BATCHED_MOE_PREFIX = "__skyrl_batched_moe_fp8__:"
SERIALIZED_WEIGHT_PREFIX = "__skyrl_serialized__:"


class WeightKind(str, Enum):
    ROUTED_EXPERT = "routed_expert"
    LINEAR = "linear"
    PASSTHROUGH = "passthrough"


class ExpertExportLayout(str, Enum):
    CHECKPOINT = "checkpoint"
    PACKED = "packed"


@dataclass(frozen=True)
class MoeProjection:
    hf_name: str
    vllm_param: str
    shard_id: str


@dataclass(frozen=True)
class MoeExpertSpec:
    source_suffix: str
    split_dim: int
    projections: tuple[MoeProjection, ...]


@dataclass(frozen=True)
class QuantizationTarget:
    checkpoint_name: str
    tensor: torch.Tensor
    kind: WeightKind
    projection: MoeProjection | None = None
    batched_experts: bool = False


@dataclass(frozen=True)
class ReceiverTensorRole:
    checkpoint_suffix: str
    parameter_suffix: str


@dataclass(frozen=True)
class ReceiverTarget:
    parameter_name: str
    shard_id: str


@dataclass(frozen=True)
class VllmSerializedWeightConfig:
    quantization: str
    quantization_config: dict[str, Any]
    load_format: str = "dummy"
    required_model_dtype: str | None = None


@dataclass(frozen=True)
class ModelQuantizationSpec:
    name: str
    model_types: frozenset[str]
    quantizable_weight_suffixes: tuple[str, ...]
    moe_expert_specs: tuple[MoeExpertSpec, ...]
    ignored_layers_fn: Callable[[Any], list[str]]
    should_quantize_fn: Callable[[str, Sequence[int]], bool] | None = None
    expert_export_layout: ExpertExportLayout = ExpertExportLayout.CHECKPOINT

    def matches(self, model_type: str) -> bool:
        return model_type in self.model_types

    def should_quantize(self, name: str, shape: Sequence[int]) -> bool:
        if self.should_quantize_fn is not None:
            return self.should_quantize_fn(name, shape)
        return name.endswith(".weight") and len(shape) == 2 and name.endswith(self.quantizable_weight_suffixes)

    def ignored_layers(self, hf_config: Any) -> list[str]:
        return self.ignored_layers_fn(hf_config)

    def moe_expert_spec(self, name: str) -> MoeExpertSpec | None:
        for spec in self.moe_expert_specs:
            if name.endswith(spec.source_suffix):
                return spec
        return None

    def normalize_moe_target(self, name: str, tensor: torch.Tensor) -> tuple[QuantizationTarget, ...] | None:
        if ".mlp.experts" not in name:
            return None
        spec = self.moe_expert_spec(name)
        if spec is None:
            return None
        if tensor.ndim != 3:
            raise ValueError(f"Batched MoE expert tensor must be 3D, got shape={tuple(tensor.shape)}")

        base = name[: -len(spec.source_suffix)]
        if len(spec.projections) == 1:
            tensors = (tensor,)
        else:
            split_dim = spec.split_dim
            split_size = tensor.shape[split_dim]
            if split_size % len(spec.projections) != 0:
                raise ValueError(
                    f"Batched MoE gate_up_proj output dimension must be even, got shape={tuple(tensor.shape)}"
                )
            tensors = tensor.chunk(len(spec.projections), dim=split_dim)

        return tuple(
            QuantizationTarget(
                checkpoint_name=f"{base}.{projection.hf_name}.weight",
                tensor=projection_tensor,
                kind=WeightKind.ROUTED_EXPERT,
                projection=projection,
                batched_experts=True,
            )
            for projection, projection_tensor in zip(spec.projections, tensors)
        )


class SerializedWeightStrategy(ABC):
    mode: str
    receiver_tensor_roles: tuple[ReceiverTensorRole, ...]
    supported_model_types: frozenset[str]
    reject_unknown_routed_experts: bool = False
    use_legacy_wire_prefix: bool = False
    uses_block_scale_runtime_contract: bool = False
    required_model_dtype: str | None = None
    vllm_quantization: str
    vllm_load_format: str = "dummy"

    def supports_model_type(self, model_type: str) -> bool:
        return model_type in self.supported_model_types

    def validate_model_type(self, model_type: str, model_path: str | None = None) -> None:
        if not self.supports_model_type(model_type):
            raise ValueError(f"{self.mode} does not support model_type={model_type!r}")

    @abstractmethod
    def supports(self, target: QuantizationTarget) -> bool:
        """Return whether this strategy quantizes a normalized target."""

    @abstractmethod
    def serialize(self, target: QuantizationTarget) -> Iterator[tuple[str, torch.Tensor]]:
        """Serialize one supported target."""

    @abstractmethod
    def vllm_config(
        self,
        inference_config: Any,
        hf_config: Any,
        model_spec: ModelQuantizationSpec,
    ) -> VllmSerializedWeightConfig:
        """Return vLLM initialization settings."""

    def wire_name(self, checkpoint_name: str) -> str:
        if self.use_legacy_wire_prefix:
            return f"{LEGACY_BATCHED_MOE_PREFIX}{checkpoint_name}"
        return f"{SERIALIZED_WEIGHT_PREFIX}{self.mode}:{checkpoint_name}"

    def receiver_target(
        self,
        checkpoint_name: str,
        model_specs: Sequence[ModelQuantizationSpec],
    ) -> ReceiverTarget | None:
        for model_spec in model_specs:
            for expert_spec in model_spec.moe_expert_specs:
                for projection in expert_spec.projections:
                    for role in self.receiver_tensor_roles:
                        suffix = f".experts.{projection.hf_name}{role.checkpoint_suffix}"
                        if checkpoint_name.endswith(suffix):
                            parameter_name = (
                                checkpoint_name[: -len(suffix)] + projection.vllm_param + role.parameter_suffix
                            )
                            return ReceiverTarget(parameter_name, projection.shard_id)
        return None


def iter_serialized_weight_tensors(
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    model_spec: ModelQuantizationSpec,
    strategy: SerializedWeightStrategy,
    *,
    model_type: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    targets = model_spec.normalize_moe_target(name, tensor)
    if targets is not None:
        for target in targets:
            if not strategy.supports(target):
                yield target.checkpoint_name, target.tensor.to(dtype=target_dtype)
                continue
            for output_name, output_tensor in strategy.serialize(target):
                yield strategy.wire_name(output_name), output_tensor
        return

    reject_unknown_experts = strategy.reject_unknown_routed_experts and getattr(strategy, "expert_only", False)
    if reject_unknown_experts and ".mlp.experts." in name and tensor.ndim >= 2:
        raise ValueError(f"Unsupported routed-expert export tensor for model_type={model_type!r}: {name}")

    kind = WeightKind.LINEAR if model_spec.should_quantize(name, tensor.shape) else WeightKind.PASSTHROUGH
    target = QuantizationTarget(checkpoint_name=name, tensor=tensor, kind=kind)
    if kind is WeightKind.LINEAR and strategy.supports(target):
        yield from strategy.serialize(target)
        return
    yield name, tensor.to(dtype=target_dtype)
