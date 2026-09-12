"""Build immutable source-to-consumer buckets for LoRA NCCL transport."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import prod
from typing import Any, Mapping

import torch

from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
    LoRAConsumerPlan,
    LoRAConsumerPull,
)


def _pull_sort_key(pull: LoRAConsumerPull) -> tuple:
    return (
        pull.source_rank,
        pull.source_slice.key,
        pull.source_slice.starts,
        pull.source_slice.stops,
    )


def _pull_elements(pull: LoRAConsumerPull) -> int:
    return prod(
        stop - start
        for start, stop in zip(
            pull.source_slice.starts,
            pull.source_slice.stops,
            strict=True,
        )
    )


@dataclass(frozen=True)
class LoRANcclConsumerRoute:
    """Serializable source pulls required by one inference rank."""

    inference_rank: int
    source_layout_digest: str
    pulls: tuple[LoRAConsumerPull, ...]

    def __post_init__(self) -> None:
        if self.inference_rank < 0:
            raise ValueError("LoRA NCCL inference ranks must be non-negative")
        if len(self.source_layout_digest) != 64:
            raise ValueError("LoRA NCCL routes require a SHA-256 layout digest")
        if not self.pulls:
            raise ValueError("LoRA NCCL routes require at least one source pull")
        canonical = tuple(sorted(self.pulls, key=_pull_sort_key))
        if self.pulls != canonical or len(self.pulls) != len(set(self.pulls)):
            raise ValueError("LoRA NCCL route pulls must be canonical and unique")

    @classmethod
    def from_consumer_plan(
        cls,
        inference_rank: int,
        plan: LoRAConsumerPlan,
    ) -> "LoRANcclConsumerRoute":
        return cls(
            inference_rank,
            plan.source_layout_digest,
            tuple(sorted(plan.pulls, key=_pull_sort_key)),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "inference_rank": self.inference_rank,
            "source_layout_digest": self.source_layout_digest,
            "pulls": [
                {
                    "source_rank": pull.source_rank,
                    "key": pull.source_slice.key,
                    "starts": pull.source_slice.starts,
                    "stops": pull.source_slice.stops,
                }
                for pull in self.pulls
            ],
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "LoRANcclConsumerRoute":
        from skyrl.backends.skyrl_train.weight_sync.lora_transport.contracts import (
            LoRASourceSlice,
        )

        return cls(
            inference_rank=int(data["inference_rank"]),
            source_layout_digest=data["source_layout_digest"],
            pulls=tuple(
                LoRAConsumerPull(
                    source_rank=int(pull["source_rank"]),
                    source_slice=LoRASourceSlice(
                        key=pull["key"],
                        starts=tuple(pull["starts"]),
                        stops=tuple(pull["stops"]),
                    ),
                )
                for pull in data["pulls"]
            ),
        )


@dataclass(frozen=True)
class LoRANcclBucket:
    """One packed FP32 transfer from a producer to one inference rank."""

    source_rank: int
    inference_rank: int
    pulls: tuple[LoRAConsumerPull, ...]

    def __post_init__(self) -> None:
        if self.source_rank < 0 or self.inference_rank < 0:
            raise ValueError("LoRA NCCL ranks must be non-negative")
        if not self.pulls:
            raise ValueError("LoRA NCCL buckets require at least one source slice")
        if any(pull.source_rank != self.source_rank for pull in self.pulls):
            raise ValueError("LoRA NCCL bucket pulls must belong to its source rank")

    @property
    def source_bytes(self) -> int:
        return torch.empty((), dtype=torch.float32).element_size() * sum(_pull_elements(pull) for pull in self.pulls)


@dataclass(frozen=True)
class LoRANcclSourceGroup:
    """One persistent communicator from a producer to its consumers."""

    source_rank: int
    inference_ranks: tuple[int, ...]
    buckets: tuple[LoRANcclBucket, ...]

    def __post_init__(self) -> None:
        if self.source_rank < 0:
            raise ValueError("LoRA NCCL source ranks must be non-negative")
        if not self.inference_ranks or self.inference_ranks != tuple(sorted(set(self.inference_ranks))):
            raise ValueError("LoRA NCCL source-group consumers must be sorted and unique")
        if not self.buckets:
            raise ValueError("LoRA NCCL source groups require at least one transfer bucket")
        if any(bucket.source_rank != self.source_rank for bucket in self.buckets):
            raise ValueError("LoRA NCCL source-group buckets must belong to its producer")
        if {bucket.inference_rank for bucket in self.buckets} != set(self.inference_ranks):
            raise ValueError("LoRA NCCL source-group buckets must cover exactly its consumers")

    @property
    def source_bytes(self) -> int:
        return sum(bucket.source_bytes for bucket in self.buckets)


@dataclass(frozen=True)
class LoRANcclEdgeReceipt:
    """Static transfer volume for one producer-consumer edge."""

    source_rank: int
    inference_rank: int
    bucket_count: int
    pull_count: int
    transmitted_bytes: int


@dataclass(frozen=True)
class LoRANcclPlanReceipt:
    """Auditable topology and byte accounting for one static plan."""

    plan_digest: str
    source_layout_digest: str
    inference_rank_count: int
    source_group_count: int
    edge_count: int
    bucket_count: int
    pull_count: int
    unique_pull_count: int
    unique_source_bytes: int
    transmitted_bytes: int
    replication_bytes: int
    maximum_bucket_bytes: int
    edges: tuple[LoRANcclEdgeReceipt, ...]


@dataclass(frozen=True)
class LoRANcclPlan:
    """Static packed routes shared by every adapter generation."""

    source_layout_digest: str
    inference_ranks: tuple[int, ...]
    packed_buffer_size_bytes: int
    source_groups: tuple[LoRANcclSourceGroup, ...]
    plan_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if len(self.source_layout_digest) != 64:
            raise ValueError("LoRA NCCL plans require a SHA-256 source layout digest")
        if not self.inference_ranks or self.inference_ranks != tuple(sorted(set(self.inference_ranks))):
            raise ValueError("LoRA NCCL inference ranks must be sorted and unique")
        if self.packed_buffer_size_bytes <= 0:
            raise ValueError("LoRA NCCL packed buffer size must be positive")
        if not self.source_groups:
            raise ValueError("LoRA NCCL plans require at least one source group")
        source_ranks = tuple(group.source_rank for group in self.source_groups)
        if source_ranks != tuple(sorted(set(source_ranks))):
            raise ValueError("LoRA NCCL source groups must be ordered and unique")
        if any(bucket.source_bytes > self.packed_buffer_size_bytes for bucket in self.buckets):
            raise ValueError("LoRA NCCL bucket exceeds the packed buffer size")
        if {bucket.inference_rank for bucket in self.buckets} != set(self.inference_ranks):
            raise ValueError("Every LoRA NCCL inference rank must receive at least one bucket")
        payload = {
            "source_layout_digest": self.source_layout_digest,
            "inference_ranks": self.inference_ranks,
            "packed_buffer_size_bytes": self.packed_buffer_size_bytes,
            "source_groups": [asdict(group) for group in self.source_groups],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        object.__setattr__(self, "plan_digest", hashlib.sha256(encoded).hexdigest())

    @property
    def source_bytes(self) -> int:
        return sum(group.source_bytes for group in self.source_groups)

    @property
    def buckets(self) -> tuple[LoRANcclBucket, ...]:
        return tuple(bucket for group in self.source_groups for bucket in group.buckets)


def build_lora_nccl_plan(
    consumer_plans: Mapping[int, LoRAConsumerPlan | LoRANcclConsumerRoute],
    packed_buffer_size_bytes: int,
) -> LoRANcclPlan:
    """Bucket exactly the slices requested by each inference rank."""
    if not consumer_plans:
        raise ValueError("LoRA NCCL planning requires at least one consumer")
    if packed_buffer_size_bytes <= 0:
        raise ValueError("LoRA NCCL packed buffer size must be positive")
    inference_ranks = tuple(sorted(consumer_plans))
    if inference_ranks[0] < 0:
        raise ValueError("LoRA NCCL inference ranks must be non-negative")
    if any(
        isinstance(route, LoRANcclConsumerRoute) and route.inference_rank != inference_rank
        for inference_rank, route in consumer_plans.items()
    ):
        raise ValueError("LoRA NCCL route rank does not match its mapping key")
    layout_digests = {plan.source_layout_digest for plan in consumer_plans.values()}
    if len(layout_digests) != 1:
        raise ValueError("LoRA NCCL consumer plans must share one source layout")

    buckets = []
    for inference_rank in inference_ranks:
        pulls_by_source: dict[int, list[LoRAConsumerPull]] = {}
        for pull in consumer_plans[inference_rank].pulls:
            pulls_by_source.setdefault(pull.source_rank, []).append(pull)
        if not pulls_by_source:
            raise ValueError(f"LoRA NCCL inference rank {inference_rank} has no source slices")
        for source_rank, pulls in sorted(pulls_by_source.items()):
            current = []
            current_bytes = 0
            for pull in sorted(pulls, key=_pull_sort_key):
                pull_bytes = torch.empty((), dtype=torch.float32).element_size() * _pull_elements(pull)
                if pull_bytes > packed_buffer_size_bytes:
                    raise ValueError(
                        f"LoRA NCCL source slice {pull.source_slice.key!r} requires "
                        f"{pull_bytes} bytes, exceeding buffer size "
                        f"{packed_buffer_size_bytes}"
                    )
                if current and current_bytes + pull_bytes > packed_buffer_size_bytes:
                    buckets.append(
                        LoRANcclBucket(
                            source_rank,
                            inference_rank,
                            tuple(current),
                        )
                    )
                    current = []
                    current_bytes = 0
                current.append(pull)
                current_bytes += pull_bytes
            if current:
                buckets.append(
                    LoRANcclBucket(
                        source_rank,
                        inference_rank,
                        tuple(current),
                    )
                )
    source_groups = tuple(
        LoRANcclSourceGroup(
            source_rank=source_rank,
            inference_ranks=tuple(
                sorted({bucket.inference_rank for bucket in buckets if bucket.source_rank == source_rank})
            ),
            buckets=tuple(bucket for bucket in buckets if bucket.source_rank == source_rank),
        )
        for source_rank in sorted({bucket.source_rank for bucket in buckets})
    )
    return LoRANcclPlan(
        source_layout_digest=next(iter(layout_digests)),
        inference_ranks=inference_ranks,
        packed_buffer_size_bytes=packed_buffer_size_bytes,
        source_groups=source_groups,
    )


def build_lora_nccl_plan_receipt(plan: LoRANcclPlan) -> LoRANcclPlanReceipt:
    """Summarize an exact plan without counting replicated pulls as unique bytes."""
    pulls = tuple(pull for bucket in plan.buckets for pull in bucket.pulls)
    unique_pulls = tuple(sorted(set(pulls), key=_pull_sort_key))
    pulls_by_source: dict[tuple[int, str], list[LoRAConsumerPull]] = {}
    for pull in unique_pulls:
        pulls_by_source.setdefault(
            (pull.source_rank, pull.source_slice.key),
            [],
        ).append(pull)
    for source_pulls in pulls_by_source.values():
        for index, pull in enumerate(source_pulls):
            for other in source_pulls[index + 1 :]:
                if all(
                    max(a, c) < min(b, d)
                    for a, b, c, d in zip(
                        pull.source_slice.starts,
                        pull.source_slice.stops,
                        other.source_slice.starts,
                        other.source_slice.stops,
                        strict=True,
                    )
                ):
                    raise ValueError("LoRA NCCL accounting does not support partially overlapping " "source slices")
    edges = tuple(
        LoRANcclEdgeReceipt(
            source_rank=group.source_rank,
            inference_rank=inference_rank,
            bucket_count=len(edge_buckets),
            pull_count=sum(len(bucket.pulls) for bucket in edge_buckets),
            transmitted_bytes=sum(bucket.source_bytes for bucket in edge_buckets),
        )
        for group in plan.source_groups
        for inference_rank in group.inference_ranks
        if (edge_buckets := tuple(bucket for bucket in group.buckets if bucket.inference_rank == inference_rank))
    )
    unique_source_bytes = 4 * sum(_pull_elements(pull) for pull in unique_pulls)
    transmitted_bytes = plan.source_bytes
    return LoRANcclPlanReceipt(
        plan_digest=plan.plan_digest,
        source_layout_digest=plan.source_layout_digest,
        inference_rank_count=len(plan.inference_ranks),
        source_group_count=len(plan.source_groups),
        edge_count=len(edges),
        bucket_count=len(plan.buckets),
        pull_count=len(pulls),
        unique_pull_count=len(unique_pulls),
        unique_source_bytes=unique_source_bytes,
        transmitted_bytes=transmitted_bytes,
        replication_bytes=transmitted_bytes - unique_source_bytes,
        maximum_bucket_bytes=max(bucket.source_bytes for bucket in plan.buckets),
        edges=edges,
    )


def pack_lora_nccl_bucket(
    bucket: LoRANcclBucket,
    source_tensors: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Copy one bucket's exact source rectangles into contiguous FP32 storage."""
    device = None
    for pull in bucket.pulls:
        source = source_tensors[pull.source_slice.key]
        if source.dtype is not torch.float32:
            raise ValueError("LoRA NCCL sources must remain FP32")
        pull.source_slice.validate_shape(tuple(source.shape))
        if device is None:
            device = source.device
        elif source.device != device:
            raise ValueError("One LoRA NCCL bucket cannot span source devices")
    packed = torch.empty(
        bucket.source_bytes // torch.empty((), dtype=torch.float32).element_size(),
        dtype=torch.float32,
        device=device,
    )
    return pack_lora_nccl_bucket_into(bucket, source_tensors, packed)


def pack_lora_nccl_bucket_into(
    bucket: LoRANcclBucket,
    source_tensors: Mapping[str, torch.Tensor],
    packed_buffer: torch.Tensor,
) -> torch.Tensor:
    """Pack one bucket into the reusable prefix of caller-owned FP32 storage."""
    if packed_buffer.dtype is not torch.float32 or packed_buffer.ndim != 1 or not packed_buffer.is_contiguous():
        raise ValueError("LoRA NCCL packed buffers must be contiguous flat FP32 tensors")
    required = bucket.source_bytes // packed_buffer.element_size()
    if packed_buffer.numel() < required:
        raise ValueError(f"LoRA NCCL packed buffer has {packed_buffer.numel()} elements, " f"requires {required}")
    packed = packed_buffer[:required]
    offset = 0
    for pull in bucket.pulls:
        source = source_tensors[pull.source_slice.key]
        if source.dtype is not torch.float32:
            raise ValueError("LoRA NCCL sources must remain FP32")
        if source.device != packed.device:
            raise ValueError("LoRA NCCL sources and packed buffer must share a device")
        pull.source_slice.validate_shape(tuple(source.shape))
        shape = tuple(
            stop - start
            for start, stop in zip(
                pull.source_slice.starts,
                pull.source_slice.stops,
                strict=True,
            )
        )
        elements = prod(shape)
        packed[offset : offset + elements].view(shape).copy_(source[pull.source_slice.indices])
        offset += elements
    return packed


def unpack_lora_nccl_bucket(
    bucket: LoRANcclBucket,
    packed: torch.Tensor,
) -> dict[LoRAConsumerPull, torch.Tensor]:
    """Expose validated FP32 views for the common BF16 assembly path."""
    if packed.dtype is not torch.float32 or packed.ndim != 1 or not packed.is_contiguous():
        raise ValueError("LoRA NCCL receive buffers must be contiguous flat FP32 tensors")
    expected = sum(_pull_elements(pull) for pull in bucket.pulls)
    if packed.numel() != expected:
        raise ValueError(f"LoRA NCCL receive buffer has {packed.numel()} elements, expected {expected}")
    output = {}
    offset = 0
    for pull in bucket.pulls:
        shape = tuple(
            stop - start
            for start, stop in zip(
                pull.source_slice.starts,
                pull.source_slice.stops,
                strict=True,
            )
        )
        elements = prod(shape)
        output[pull] = packed[offset : offset + elements].view(shape)
        offset += elements
    return output
