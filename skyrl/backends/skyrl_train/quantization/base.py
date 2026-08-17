"""Shared model-layout and quantization-strategy interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Any

import torch

SERIALIZED_WEIGHT_PREFIX = "__skyrl_serialized__:"


class ExpertExportLayout(str, Enum):
    """Expert tensor layout requested from Megatron-Bridge."""

    CHECKPOINT = "checkpoint"
    PACKED = "packed"


@dataclass(frozen=True)
class QuantizationTarget:
    """One independently selectable group of model weights."""

    name: str
    matches_exported_weight: Callable[[str, Sequence[int]], bool]
    megatron_patterns: tuple[str, ...] = ()
    checkpoint_component: str | None = None
    vllm_parameter: str | None = None
    vllm_shard_id: str | None = None


@dataclass(frozen=True)
class ExpertWeight:
    """One logical expert weight produced from a Bridge export tensor."""

    checkpoint_component: str
    target: str


@dataclass(frozen=True)
class MoeExpertSpec:
    """How a packed expert tensor is split into logical checkpoint weights."""

    source_suffix: str
    split_dim: int | None
    output_weights: tuple[ExpertWeight, ...]


@dataclass(frozen=True)
class ModelQuantizationLayout:
    """Model-specific target, naming, splitting, and receiver metadata."""

    name: str
    model_types: frozenset[str]
    targets: tuple[QuantizationTarget, ...]
    moe_expert_specs: tuple[MoeExpertSpec, ...] = ()
    ignored_layers: Callable[[Any], list[str]] = lambda _config: []
    expert_export_layout: ExpertExportLayout = ExpertExportLayout.CHECKPOINT

    def matches(self, model_type: str) -> bool:
        return model_type in self.model_types

    def target(self, name: str) -> QuantizationTarget:
        for target in self.targets:
            if target.name == name:
                return target
        raise ValueError(f"Model layout {self.name!r} has no quantization target {name!r}")

    def validate_targets(self, target_names: Sequence[str]) -> None:
        available = {target.name for target in self.targets}
        unknown = set(target_names) - available
        if unknown:
            raise ValueError(
                f"Quantization targets {sorted(unknown)} are unavailable for layout {self.name!r}; "
                f"available targets: {sorted(available)}"
            )

    def split_exported_weight(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        """Split a packed Bridge tensor into logical checkpoint weights."""

        spec = next(
            (spec for spec in self.moe_expert_specs if name.endswith(spec.source_suffix)),
            None,
        )
        if spec is None:
            return ((name, tensor),)
        if tensor.ndim != 3:
            raise ValueError(f"Batched MoE expert tensor must be 3D, got shape={tuple(tensor.shape)}")

        base = name[: -len(spec.source_suffix)]
        if len(spec.output_weights) == 1:
            tensors = (tensor,)
        else:
            assert spec.split_dim is not None
            if tensor.shape[spec.split_dim] % len(spec.output_weights) != 0:
                raise ValueError(
                    f"Packed expert dimension must be divisible by {len(spec.output_weights)} "
                    f"for {spec.source_suffix}, got shape={tuple(tensor.shape)}"
                )
            tensors = tensor.chunk(len(spec.output_weights), dim=spec.split_dim)

        return tuple(
            (f"{base}.{weight.checkpoint_component}.weight", output_tensor)
            for weight, output_tensor in zip(spec.output_weights, tensors)
        )

    def weight_target(self, name: str, shape: Sequence[int]) -> str | None:
        """Classify an exported checkpoint weight into a named target."""

        return self._weight_target(name, tuple(shape))

    @cache
    def _weight_target(self, name: str, shape: tuple[int, ...]) -> str | None:
        matches = [target.name for target in self.targets if target.matches_exported_weight(name, shape)]
        if len(matches) > 1:
            raise ValueError(f"Weight {name!r} matches multiple targets in layout {self.name!r}: {matches}")
        return matches[0] if matches else None


class QuantizationStrategy(ABC):
    """Numerical format applied to selected model-layout targets."""

    mode: str
    target_names: frozenset[str]
    receiver_suffixes: Mapping[str, str]
    reject_unknown_routed_experts: bool = False
    required_model_dtype: str | None = None
    vllm_quantization: str
    vllm_load_format: str = "dummy"

    def validate_layout(self, layout: ModelQuantizationLayout) -> None:
        layout.validate_targets(self.target_names)

    def should_quantize(self, target: str | None, name: str, shape: Sequence[int]) -> bool:
        del name, shape
        return target in self.target_names

    def build_te_recipe(self, *, persistent: bool) -> dict:
        raise NotImplementedError(f"{self.mode} does not support Megatron training")

    def configure_megatron_provider(self, provider) -> None:
        raise NotImplementedError(f"{self.mode} does not support Megatron training")

    def build_runtime_env(self) -> dict[str, str]:
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
        layout: ModelQuantizationLayout,
    ) -> dict[str, Any]:
        """Return vLLM's format-specific quantization configuration."""

    def encode_serialized_name(self, checkpoint_name: str) -> str:
        return f"{SERIALIZED_WEIGHT_PREFIX}{self.mode}:{checkpoint_name}"


def decode_serialized_name(name: str) -> tuple[str, str] | None:
    """Decode a strategy-tagged checkpoint name."""

    if not name.startswith(SERIALIZED_WEIGHT_PREFIX):
        return None
    mode, separator, checkpoint_name = name.removeprefix(SERIALIZED_WEIGHT_PREFIX).partition(":")
    if not separator or not mode or not checkpoint_name:
        raise ValueError(f"Malformed serialized weight name {name!r}")
    return mode, checkpoint_name


def iter_serialized_weight_tensors(
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    layout: ModelQuantizationLayout,
    strategy: QuantizationStrategy,
    *,
    model_type: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Classify and serialize one Megatron-Bridge export tensor."""

    for output_name, output_tensor in layout.split_exported_weight(name, tensor):
        target = layout.weight_target(output_name, output_tensor.shape)
        if strategy.should_quantize(target, output_name, output_tensor.shape):
            for serialized_name, serialized_tensor in strategy.serialize_weight(
                output_name,
                output_tensor,
                batched_experts=output_tensor.ndim == 3,
            ):
                yield (
                    strategy.encode_serialized_name(serialized_name) if output_tensor.ndim == 3 else serialized_name,
                    serialized_tensor,
                )
            continue

        if (
            strategy.reject_unknown_routed_experts
            and ".mlp.experts." in output_name
            and output_tensor.ndim >= 2
            and target is None
        ):
            raise ValueError(f"Unsupported routed-expert export tensor for model_type={model_type!r}: {output_name}")
        yield output_name, output_tensor.to(dtype=target_dtype)
