"""Shared quantized-model layout and strategy interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeAlias

import torch

SERIALIZED_WEIGHT_PREFIX = "__skyrl_serialized__:"

WeightCategory: TypeAlias = Literal[
    "linear",
    "routed_expert_gate",
    "routed_expert_up",
    "routed_expert_down",
]


class ExpertExportLayout(str, Enum):
    """Expert tensor layout requested from Megatron-Bridge."""

    CHECKPOINT = "checkpoint"
    PACKED = "packed"


@dataclass(frozen=True)
class ExpertWeight:
    """One logical expert weight produced from a Bridge export tensor."""

    checkpoint_component: str
    category: WeightCategory


@dataclass(frozen=True)
class MoeExpertSpec:
    """How a packed expert tensor is split into logical checkpoint weights."""

    source_suffix: str
    split_dim: int
    output_weights: tuple[ExpertWeight, ...]


@dataclass(frozen=True)
class QuantizedModelLayout:
    """Model-specific weight naming, splitting, and classification."""

    name: str
    model_types: frozenset[str]
    quantizable_weight_suffixes: tuple[str, ...]
    moe_expert_specs: tuple[MoeExpertSpec, ...]
    expert_export_layout: ExpertExportLayout = ExpertExportLayout.CHECKPOINT

    def matches(self, model_type: str) -> bool:
        """Return whether this layout describes the model type."""

        return model_type in self.model_types

    def split_exported_weight(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        """Split a packed Bridge tensor into logical checkpoint weights."""

        if ".mlp.experts" not in name:
            return ((name, tensor),)
        spec = next((spec for spec in self.moe_expert_specs if name.endswith(spec.source_suffix)), None)
        if spec is None:
            return ((name, tensor),)
        if tensor.ndim != 3:
            raise ValueError(f"Batched MoE expert tensor must be 3D, got shape={tuple(tensor.shape)}")

        base = name[: -len(spec.source_suffix)]
        if len(spec.output_weights) == 1:
            tensors = (tensor,)
        else:
            split_size = tensor.shape[spec.split_dim]
            if split_size % len(spec.output_weights) != 0:
                raise ValueError(
                    f"Batched MoE gate_up_proj output dimension must be even, got shape={tuple(tensor.shape)}"
                )
            tensors = tensor.chunk(len(spec.output_weights), dim=spec.split_dim)

        return tuple(
            (f"{base}.{weight.checkpoint_component}.weight", output_tensor)
            for weight, output_tensor in zip(spec.output_weights, tensors)
        )

    def weight_category(self, name: str, shape: Sequence[int]) -> WeightCategory | None:
        """Classify a logical checkpoint weight."""

        if ".mlp.experts." in name:
            for spec in self.moe_expert_specs:
                for weight in spec.output_weights:
                    if name.endswith(f".{weight.checkpoint_component}.weight"):
                        return weight.category
        if name.endswith(".weight") and len(shape) == 2 and name.endswith(self.quantizable_weight_suffixes):
            return "linear"
        return None


class QuantizationStrategy(ABC):
    """Cross-stack quantization format and model scope."""

    mode: str
    quantized_categories: frozenset[WeightCategory]
    receiver_suffixes: Mapping[str, str]
    supported_model_types: frozenset[str]
    reject_unknown_routed_experts: bool = False
    required_model_dtype: str | None = None
    vllm_quantization: str
    vllm_load_format: str = "dummy"

    def supports_model_type(self, model_type: str) -> bool:
        """Return whether the format supports the model type."""

        return model_type in self.supported_model_types

    def validate_model_type(self, model_type: str, model_path: str | None = None) -> None:
        """Reject model layouts unsupported by the format."""

        if not self.supports_model_type(model_type):
            raise ValueError(f"{self.mode} does not support model_type={model_type!r}")

    def should_quantize(
        self,
        category: WeightCategory | None,
        name: str,
        shape: Sequence[int],
    ) -> bool:
        """Return whether a classified weight should use this strategy."""

        del name, shape
        return category in self.quantized_categories

    def build_te_recipe(self, *, persistent: bool) -> dict:
        """Build Transformer Engine settings for quantized modules."""

        raise NotImplementedError(f"{self.mode} does not support Megatron training")

    def configure_megatron_provider(self, provider) -> None:
        """Set provider fields required by the strategy."""

        raise NotImplementedError(f"{self.mode} does not support Megatron training")

    def build_runtime_env(self) -> dict[str, str]:
        """Return process-level environment required by the strategy."""

        return {}

    @abstractmethod
    def serialize_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        batched_experts: bool,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield encoded weight and companion tensors."""

    @abstractmethod
    def vllm_quantization_config(
        self,
        inference_config: Any,
        hf_config: Any,
        layout: QuantizedModelLayout,
    ) -> dict[str, Any]:
        """Return vLLM's format-specific quantization configuration."""

    def encode_serialized_name(self, checkpoint_name: str) -> str:
        """Add the strategy mode used by the inference receiver."""

        return f"{SERIALIZED_WEIGHT_PREFIX}{self.mode}:{checkpoint_name}"


def iter_serialized_weight_tensors(
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    layout: QuantizedModelLayout,
    strategy: QuantizationStrategy,
    *,
    model_type: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Classify and serialize one Megatron-Bridge export tensor."""

    for output_name, output_tensor in layout.split_exported_weight(name, tensor):
        category = layout.weight_category(output_name, output_tensor.shape)
        if strategy.should_quantize(category, output_name, output_tensor.shape):
            for serialized_name, serialized_tensor in strategy.serialize_weight(
                output_name,
                output_tensor,
                batched_experts=output_tensor.ndim == 3,
            ):
                yield (
                    strategy.encode_serialized_name(serialized_name)
                    if output_tensor.ndim == 3
                    else serialized_name,
                    serialized_tensor,
                )
            continue

        if (
            strategy.reject_unknown_routed_experts
            and ".mlp.experts." in output_name
            and output_tensor.ndim >= 2
            and category is None
        ):
            raise ValueError(f"Unsupported routed-expert export tensor for model_type={model_type!r}: {output_name}")
        yield output_name, output_tensor.to(dtype=target_dtype)
