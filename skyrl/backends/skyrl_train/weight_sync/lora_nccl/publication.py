"""Trainer-side fixed-layout publication planning for LoRA NCCL."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from skyrl.backends.skyrl_train.weight_sync.lora_transport.bridge_sources import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    extract_lora_bridge_sources,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.contracts import (
    LoRAUpdateRequest,
)


@dataclass(frozen=True)
class LoRANcclPublication:
    """One attempt's rank-local CUDA tensors and immutable request."""

    local_tensors: dict[str, torch.Tensor]
    layout: LoRABridgeSourceLayout
    request: LoRAUpdateRequest


class LoRANcclPublicationPlanner:
    """Validate each update against one global rank-local Bridge layout."""

    def __init__(self, adapter_name: str, source_rank: int) -> None:
        if not adapter_name:
            raise ValueError("LoRA NCCL publications require an adapter name")
        if source_rank < 0:
            raise ValueError("LoRA NCCL source ranks must be non-negative")
        self._adapter_name = adapter_name
        self._source_rank = source_rank
        self._layout: LoRABridgeSourceLayout | None = None
        self._generation = -1

    def plan(
        self,
        records: Iterable[Any],
        gathered_sources: Sequence[Sequence[LoRABridgeSource]] | None = None,
    ) -> LoRANcclPublication:
        """Build the next attempt and advance generation even after a failure."""
        local_tensors, local_sources = extract_lora_bridge_sources(
            records,
            self._source_rank,
        )
        if self._layout is None:
            if gathered_sources is None:
                raise ValueError(
                    "LoRA NCCL first publication requires global source metadata"
                )
            sources = tuple(
                sorted(
                    (
                        source
                        for rank_sources in gathered_sources
                        for source in rank_sources
                    ),
                    key=lambda source: (
                        source.key,
                        source.expert_parallel_rank,
                        source.tensor_parallel_rank,
                        source.source_rank,
                    ),
                )
            )
            self._layout = LoRABridgeSourceLayout(self._adapter_name, sources)
        elif gathered_sources is not None:
            raise ValueError("LoRA NCCL source metadata is already initialized")

        expected = tuple(
            source
            for source in self._layout.sources
            if source.source_rank == self._source_rank
        )
        if local_sources != expected:
            raise ValueError(
                f"LoRA NCCL adapter {self._adapter_name!r} changed its fixed "
                "local source layout"
            )
        self._generation += 1
        return LoRANcclPublication(
            local_tensors,
            self._layout,
            LoRAUpdateRequest.from_layout(self._layout, self._generation),
        )
