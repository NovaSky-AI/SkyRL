"""Persistent packed point-to-point transport for rank-local LoRA factors."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

import torch

from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
    LoRAConsumerPlan,
    LoRAConsumerPull,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.contracts import (
    LoRAUpdateRequest,
)

from .plan import (
    LoRANcclConsumerRoute,
    LoRANcclSourceGroup,
    build_lora_nccl_plan,
    pack_lora_nccl_bucket_into,
    unpack_lora_nccl_bucket,
)


class LoRANcclCommunicator(Protocol):
    """Subset of vLLM's PyNcclCommunicator used by LoRA transport."""

    def send(self, tensor: torch.Tensor, dst: int, stream: Any = None) -> None: ...

    def recv(self, tensor: torch.Tensor, src: int, stream: Any = None) -> None: ...

    def destroy(self) -> None: ...


@dataclass(frozen=True)
class LoRANcclTransferReceipt:
    """One rank's measured packed transport envelope."""

    generation: int
    plan_digest: str
    direction: Literal["send", "receive"]
    rank: int
    bucket_count: int
    fp32_bytes: int
    envelope_seconds: float


class LoRANcclSourceSession:
    """Reuse one trainer-owned communicator and packed buffer across updates."""

    def __init__(
        self,
        source_group: LoRANcclSourceGroup,
        plan_digest: str,
        source_layout_digest: str,
        communicator: LoRANcclCommunicator,
        peer_by_inference_rank: Mapping[int, int],
        device: torch.device | str,
    ) -> None:
        _validate_digest(plan_digest, "plan")
        _validate_digest(source_layout_digest, "source layout")
        self.source_group = source_group
        self.plan_digest = plan_digest
        self.source_layout_digest = source_layout_digest
        self.communicator = communicator
        if set(peer_by_inference_rank) != set(source_group.inference_ranks):
            raise ValueError(
                "LoRA NCCL source peers do not match the source-group consumers"
            )
        self._peer_by_inference_rank = dict(peer_by_inference_rank)
        maximum_elements = (
            max(bucket.source_bytes for bucket in source_group.buckets) // 4
        )
        self._buffer = torch.empty(
            maximum_elements,
            dtype=torch.float32,
            device=device,
        )
        self._usable = True
        self._closed = False

    def send(
        self,
        request: LoRAUpdateRequest,
        source_tensors: Mapping[str, torch.Tensor],
    ) -> LoRANcclTransferReceipt:
        """Pack and send one complete source rank's contribution."""
        self._validate_request(request)
        started = time.perf_counter()
        try:
            for bucket in self.source_group.buckets:
                packed = pack_lora_nccl_bucket_into(
                    bucket,
                    source_tensors,
                    self._buffer,
                )
                self.communicator.send(
                    packed,
                    dst=self._peer_by_inference_rank[bucket.inference_rank],
                )
            _synchronize(self._buffer.device)
        except BaseException:
            self.close()
            raise
        return LoRANcclTransferReceipt(
            generation=request.generation,
            plan_digest=self.plan_digest,
            direction="send",
            rank=self.source_group.source_rank,
            bucket_count=len(self.source_group.buckets),
            fp32_bytes=self.source_group.source_bytes,
            envelope_seconds=time.perf_counter() - started,
        )

    def close(self) -> None:
        """Destroy the communicator once and release its reusable buffer."""
        if self._closed:
            return
        self.communicator.destroy()
        self._usable = False
        self._closed = True
        self._buffer = torch.empty(0, dtype=torch.float32, device=self._buffer.device)

    def _validate_request(self, request: LoRAUpdateRequest) -> None:
        if not self._usable:
            raise RuntimeError("LoRA NCCL source session is unusable")
        if request.layout_digest != self.source_layout_digest:
            raise ValueError("LoRA NCCL request changed the source layout")
        if request.source_dtype != "float32":
            raise ValueError("LoRA NCCL transport requires FP32 sources")


class LoRANcclReceiverSession:
    """Receive packed source buckets directly into independent BF16 factors."""

    def __init__(
        self,
        consumer_plan: LoRAConsumerPlan,
        inference_rank: int,
        plan_digest: str,
        packed_buffer_size_bytes: int,
        communicator: LoRANcclCommunicator,
        peer_by_source_rank: Mapping[int, int],
        device: torch.device | str,
    ) -> None:
        _validate_digest(plan_digest, "plan")
        route = LoRANcclConsumerRoute.from_consumer_plan(
            inference_rank,
            consumer_plan,
        )
        local_plan = build_lora_nccl_plan(
            {inference_rank: route},
            packed_buffer_size_bytes,
        )
        source_ranks = {bucket.source_rank for bucket in local_plan.buckets}
        if set(peer_by_source_rank) != source_ranks:
            raise ValueError("LoRA NCCL receiver peers do not match its source ranks")
        self.consumer_plan = consumer_plan
        self.inference_rank = inference_rank
        self.plan_digest = plan_digest
        self.communicator = communicator
        self._peer_by_source_rank = dict(peer_by_source_rank)
        self._buckets = local_plan.buckets
        maximum_elements = max(bucket.source_bytes for bucket in self._buckets) // 4
        self._buffer = torch.empty(
            maximum_elements,
            dtype=torch.float32,
            device=device,
        )
        self._usable = True
        self._closed = False

    def receive(
        self,
        request: LoRAUpdateRequest,
    ) -> tuple[
        dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]],
        LoRANcclTransferReceipt,
    ]:
        """Receive one generation and materialize its local BF16 staging factors."""
        self._validate_request(request)
        assembler = _LoRAConsumerAssembler(self.consumer_plan, self._buffer.device)
        started = time.perf_counter()
        try:
            for bucket in self._buckets:
                elements = bucket.source_bytes // self._buffer.element_size()
                packed = self._buffer[:elements]
                self.communicator.recv(
                    packed,
                    src=self._peer_by_source_rank[bucket.source_rank],
                )
                for pull, tensor in unpack_lora_nccl_bucket(bucket, packed).items():
                    assembler.copy(pull, tensor)
            factors = assembler.finish()
            _synchronize(self._buffer.device)
        except BaseException:
            self.close()
            raise
        receipt = LoRANcclTransferReceipt(
            generation=request.generation,
            plan_digest=self.plan_digest,
            direction="receive",
            rank=self.inference_rank,
            bucket_count=len(self._buckets),
            fp32_bytes=sum(bucket.source_bytes for bucket in self._buckets),
            envelope_seconds=time.perf_counter() - started,
        )
        return factors, receipt

    def close(self) -> None:
        """Destroy all source communicators once and release receive storage."""
        if self._closed:
            return
        self.communicator.destroy()
        self._usable = False
        self._closed = True
        self._buffer = torch.empty(0, dtype=torch.float32, device=self._buffer.device)

    def _validate_request(self, request: LoRAUpdateRequest) -> None:
        if not self._usable:
            raise RuntimeError("LoRA NCCL receiver session is unusable")
        if request.layout_digest != self.consumer_plan.source_layout_digest:
            raise ValueError("LoRA NCCL request changed the consumer layout")
        if request.source_dtype != "float32":
            raise ValueError("LoRA NCCL transport requires FP32 sources")


class _LoRAConsumerAssembler:
    """Copy streaming FP32 pulls into their final independent BF16 factors."""

    def __init__(
        self,
        plan: LoRAConsumerPlan,
        device: torch.device | str,
    ) -> None:
        self._plan = plan
        self._required = set(plan.pulls)
        self._received: set[LoRAConsumerPull] = set()
        self._copies: dict[LoRAConsumerPull, list[Any]] = {
            pull: [] for pull in plan.pulls
        }
        for copy in plan.copies:
            self._copies[plan.pulls[copy.pull_index]].append(copy)
        self._factors = {
            module.module_name: (
                [
                    torch.empty(pair[0], dtype=torch.bfloat16, device=device)
                    for pair in module.factor_shapes
                ],
                [
                    torch.empty(pair[1], dtype=torch.bfloat16, device=device)
                    for pair in module.factor_shapes
                ],
            )
            for module in plan.receiver_plan.modules
        }

    def copy(self, pull: LoRAConsumerPull, tensor: torch.Tensor) -> None:
        """Validate and copy one received source slice exactly once."""
        if pull not in self._required:
            raise ValueError("LoRA NCCL received an unplanned source slice")
        if pull in self._received:
            raise ValueError("LoRA NCCL received a source slice more than once")
        expected_shape = tuple(
            stop - start
            for start, stop in zip(
                pull.source_slice.starts,
                pull.source_slice.stops,
                strict=True,
            )
        )
        if tensor.dtype is not torch.float32 or tuple(tensor.shape) != expected_shape:
            raise ValueError("LoRA NCCL source slices must preserve FP32 shape")
        for copy in self._copies[pull]:
            shape = tuple(
                stop - start
                for start, stop in zip(copy.starts, copy.stops, strict=True)
            )
            destination = self._factors[copy.module_name][copy.component][
                copy.factor_index
            ]
            destination[
                tuple(
                    slice(start, stop)
                    for start, stop in zip(copy.starts, copy.stops, strict=True)
                )
            ].copy_(tensor.reshape(shape))
        self._received.add(pull)

    def finish(
        self,
    ) -> dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Return complete factors only after every planned pull arrived."""
        if self._received != self._required:
            raise ValueError("LoRA NCCL did not receive the complete consumer plan")
        return self._factors


def _validate_digest(digest: str, label: str) -> None:
    if len(digest) != 64:
        raise ValueError(f"LoRA NCCL {label} requires a SHA-256 digest")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
