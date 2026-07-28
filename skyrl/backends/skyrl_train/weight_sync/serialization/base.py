"""Shared abstractions for serialized weight quantization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

LEGACY_BATCHED_MOE_PREFIX = "__skyrl_batched_moe_fp8__:"
SERIALIZED_WEIGHT_PREFIX = "__skyrl_serialized__:"


class ExpertExportLayout(str, Enum):
    CHECKPOINT = "checkpoint"
    PACKED = "packed"


@dataclass(frozen=True)
class ExpertWeightMapping:
    checkpoint_component: str
    runtime_parameter_suffix: str
    shard_id: str


@dataclass(frozen=True)
class MoeExpertSpec:
    source_suffix: str
    split_dim: int
    output_weights: tuple[ExpertWeightMapping, ...]


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

    def normalize_moe_targets(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> tuple[tuple[str, torch.Tensor], ...] | None:
        if ".mlp.experts" not in name:
            return None
        spec = self.moe_expert_spec(name)
        if spec is None:
            return None
        if tensor.ndim != 3:
            raise ValueError(f"Batched MoE expert tensor must be 3D, got shape={tuple(tensor.shape)}")

        base = name[: -len(spec.source_suffix)]
        if len(spec.output_weights) == 1:
            tensors = (tensor,)
        else:
            split_dim = spec.split_dim
            split_size = tensor.shape[split_dim]
            if split_size % len(spec.output_weights) != 0:
                raise ValueError(
                    f"Batched MoE gate_up_proj output dimension must be even, got shape={tuple(tensor.shape)}"
                )
            tensors = tensor.chunk(len(spec.output_weights), dim=split_dim)

        return tuple(
            (f"{base}.{mapping.checkpoint_component}.weight", output_tensor)
            for mapping, output_tensor in zip(spec.output_weights, tensors)
        )


class SerializedWeightStrategy(ABC):
    mode: str
    receiver_suffixes: Mapping[str, str]
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
    def supports(self, name: str, tensor: torch.Tensor, *, routed_expert: bool) -> bool:
        """Return whether this strategy quantizes a normalized target."""

    @abstractmethod
    def serialize(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        batched_experts: bool,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Serialize one supported target."""

    @abstractmethod
    def vllm_quantization_config(
        self,
        inference_config: Any,
        hf_config: Any,
        model_spec: ModelQuantizationSpec,
    ) -> dict[str, Any]:
        """Return vLLM's quantization configuration."""

    def wire_name(self, checkpoint_name: str) -> str:
        if self.use_legacy_wire_prefix:
            return f"{LEGACY_BATCHED_MOE_PREFIX}{checkpoint_name}"
        return f"{SERIALIZED_WEIGHT_PREFIX}{self.mode}:{checkpoint_name}"

    def receiver_target(
        self,
        checkpoint_name: str,
        model_specs: Sequence[ModelQuantizationSpec],
    ) -> tuple[str, str] | None:
        for model_spec in model_specs:
            for expert_spec in model_spec.moe_expert_specs:
                for mapping in expert_spec.output_weights:
                    for checkpoint_suffix, parameter_suffix in self.receiver_suffixes.items():
                        suffix = f".experts.{mapping.checkpoint_component}{checkpoint_suffix}"
                        if checkpoint_name.endswith(suffix):
                            parameter_name = (
                                checkpoint_name[: -len(suffix)]
                                + mapping.runtime_parameter_suffix
                                + parameter_suffix
                            )
                            return parameter_name, mapping.shard_id
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
    targets = model_spec.normalize_moe_targets(name, tensor)
    if targets is not None:
        for target_name, target_tensor in targets:
            if not strategy.supports(target_name, target_tensor, routed_expert=True):
                yield target_name, target_tensor.to(dtype=target_dtype)
                continue
            for output_name, output_tensor in strategy.serialize(
                target_name,
                target_tensor,
                batched_experts=True,
            ):
                yield strategy.wire_name(output_name), output_tensor
        return

    reject_unknown_experts = strategy.reject_unknown_routed_experts and getattr(strategy, "expert_only", False)
    if reject_unknown_experts and ".mlp.experts." in name and tensor.ndim >= 2:
        raise ValueError(f"Unsupported routed-expert export tensor for model_type={model_type!r}: {name}")

    if model_spec.should_quantize(name, tensor.shape) and strategy.supports(name, tensor, routed_expert=False):
        yield from strategy.serialize(name, tensor, batched_experts=False)
        return
    yield name, tensor.to(dtype=target_dtype)
