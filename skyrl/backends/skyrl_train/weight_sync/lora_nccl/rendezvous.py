"""Static rendezvous for one persistent cross-node LoRA NCCL group."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
    LoRAConsumerPlan,
)

from .plan import LoRANcclPlan, LoRANcclSourceGroup
from .transport import (
    LoRANcclCommunicator,
    LoRANcclReceiverSession,
    LoRANcclSourceSession,
)

LoRANcclCommunicatorFactory = Callable[
    [str, int, int, int, torch.device | str], LoRANcclCommunicator
]


@dataclass(frozen=True)
class LoRANcclRendezvous:
    """Address and stable ranks for one cached cross-node communicator."""

    adapter_name: str
    source_layout_digest: str
    plan_digest: str
    packed_buffer_size_bytes: int
    source_ranks: tuple[int, ...]
    inference_ranks: tuple[int, ...]
    master_address: str
    master_port: int

    def __post_init__(self) -> None:
        if not self.adapter_name:
            raise ValueError("LoRA NCCL rendezvous requires an adapter name")
        for value, label in (
            (self.source_layout_digest, "source layout"),
            (self.plan_digest, "plan"),
        ):
            if len(value) != 64:
                raise ValueError(f"LoRA NCCL {label} requires a SHA-256 digest")
        if self.packed_buffer_size_bytes <= 0:
            raise ValueError("LoRA NCCL packed buffer size must be positive")
        for ranks, label in (
            (self.source_ranks, "source"),
            (self.inference_ranks, "inference"),
        ):
            if not ranks or ranks != tuple(sorted(set(ranks))):
                raise ValueError(
                    f"LoRA NCCL {label} ranks must be sorted, unique, and nonempty"
                )
            if ranks[0] < 0:
                raise ValueError(f"LoRA NCCL {label} ranks must be non-negative")
        if not self.master_address:
            raise ValueError("LoRA NCCL rendezvous requires a master address")
        if not 0 < self.master_port < 65536:
            raise ValueError("LoRA NCCL rendezvous requires a valid master port")

    @property
    def world_size(self) -> int:
        """Return every producer and consumer in the shared communicator."""
        return len(self.source_ranks) + len(self.inference_ranks)

    def get_source_peer_rank(self, source_rank: int) -> int:
        """Map a trainer rank to its dense communicator rank."""
        try:
            return self.source_ranks.index(source_rank)
        except ValueError as error:
            raise ValueError(
                f"Trainer rank {source_rank} is not in the LoRA NCCL group"
            ) from error

    def get_inference_peer_rank(self, inference_rank: int) -> int:
        """Map an inference rank to its dense communicator rank."""
        try:
            return len(self.source_ranks) + self.inference_ranks.index(inference_rank)
        except ValueError as error:
            raise ValueError(
                f"Inference rank {inference_rank} is not in the LoRA NCCL group"
            ) from error

    @classmethod
    def from_plan(
        cls,
        plan: LoRANcclPlan,
        adapter_name: str,
        master_address: str,
        master_port: int,
    ) -> "LoRANcclRendezvous":
        """Bind a validated static plan to one shared endpoint."""
        return cls(
            adapter_name=adapter_name,
            source_layout_digest=plan.source_layout_digest,
            plan_digest=plan.plan_digest,
            packed_buffer_size_bytes=plan.packed_buffer_size_bytes,
            source_ranks=tuple(group.source_rank for group in plan.source_groups),
            inference_ranks=plan.inference_ranks,
            master_address=master_address,
            master_port=master_port,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-safe static control-plane metadata."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "LoRANcclRendezvous":
        """Reconstruct and validate static control-plane metadata."""
        return cls(
            adapter_name=data["adapter_name"],
            source_layout_digest=data["source_layout_digest"],
            plan_digest=data["plan_digest"],
            packed_buffer_size_bytes=int(data["packed_buffer_size_bytes"]),
            source_ranks=tuple(data["source_ranks"]),
            inference_ranks=tuple(data["inference_ranks"]),
            master_address=data["master_address"],
            master_port=int(data["master_port"]),
        )


def open_lora_nccl_source_session(
    source_group: LoRANcclSourceGroup,
    rendezvous: LoRANcclRendezvous,
    device: torch.device | str,
    communicator_factory: LoRANcclCommunicatorFactory | None = None,
) -> LoRANcclSourceSession:
    """Join one trainer rank to the shared persistent communicator."""
    if source_group.source_rank not in rendezvous.source_ranks:
        raise ValueError("LoRA NCCL rendezvous is missing the source rank")
    if not set(source_group.inference_ranks).issubset(rendezvous.inference_ranks):
        raise ValueError("LoRA NCCL rendezvous changed source-group consumers")
    factory = communicator_factory or _open_stateless_group
    communicator = factory(
        rendezvous.master_address,
        rendezvous.master_port,
        rendezvous.get_source_peer_rank(source_group.source_rank),
        rendezvous.world_size,
        device,
    )
    return LoRANcclSourceSession(
        source_group,
        rendezvous.plan_digest,
        rendezvous.source_layout_digest,
        communicator,
        {
            rank: rendezvous.get_inference_peer_rank(rank)
            for rank in source_group.inference_ranks
        },
        device,
    )


def open_lora_nccl_receiver_session(
    consumer_plan: LoRAConsumerPlan,
    inference_rank: int,
    rendezvous: LoRANcclRendezvous,
    device: torch.device | str,
    communicator_factory: LoRANcclCommunicatorFactory | None = None,
) -> LoRANcclReceiverSession:
    """Join one inference rank to the shared persistent communicator."""
    if consumer_plan.source_layout_digest != rendezvous.source_layout_digest:
        raise ValueError("LoRA NCCL consumer layout changed before rendezvous")
    source_ranks = {pull.source_rank for pull in consumer_plan.pulls}
    if not source_ranks.issubset(rendezvous.source_ranks):
        raise ValueError("LoRA NCCL rendezvous is missing a required source rank")
    factory = communicator_factory or _open_stateless_group
    communicator = factory(
        rendezvous.master_address,
        rendezvous.master_port,
        rendezvous.get_inference_peer_rank(inference_rank),
        rendezvous.world_size,
        device,
    )
    try:
        return LoRANcclReceiverSession(
            consumer_plan,
            inference_rank,
            rendezvous.plan_digest,
            rendezvous.packed_buffer_size_bytes,
            communicator,
            {rank: rendezvous.get_source_peer_rank(rank) for rank in source_ranks},
            device,
        )
    except BaseException:
        communicator.destroy()
        raise


def _open_stateless_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: torch.device | str,
) -> LoRANcclCommunicator:
    from vllm.distributed.weight_transfer.nccl_common import (
        stateless_init_process_group,
    )

    return stateless_init_process_group(
        master_address,
        master_port,
        rank,
        world_size,
        device,
    )
