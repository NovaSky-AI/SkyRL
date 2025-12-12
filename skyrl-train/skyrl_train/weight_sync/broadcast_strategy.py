"""Broadcast-based weight transfer strategy using torch.distributed.

This module implements the broadcast transfer strategy for synchronizing model weights
from training workers to inference engines using NCCL/Gloo broadcast operations.
"""

import asyncio
from typing import Iterable, Iterator, Tuple

import torch

from skyrl_train.inference_engines.base import NamedWeightsUpdateRequest
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.weight_sync.base import WeightChunk
from skyrl_train.weight_sync.transfer_strategy import (
    WeightTransferStrategy,
    WeightTransferSender,
    WeightTransferReceiver,
)


class BroadcastWeightTransferSender(WeightTransferSender):
    """Sends weights via torch.distributed.broadcast.

    The sender broadcasts tensors from rank 0 to all other ranks in the
    model update group, while coordinating with inference engines via RPC.
    """

    def __init__(
        self,
        rank: int,
        model_update_group: torch.distributed.ProcessGroup,
        model_dtype_str: str,
        inference_client: InferenceEngineClient,
    ) -> None:
        """Initialize the broadcast sender.

        Args:
            rank: This worker's rank in the distributed group.
            model_update_group: Process group for broadcast operations.
            model_dtype_str: Model dtype as string (e.g., "bfloat16") for request metadata.
            inference_client: Client for coordinating with inference engines.
        """
        self._rank = rank
        self._model_update_group = model_update_group
        self._model_dtype_str = model_dtype_str
        self._inference_client = inference_client

    async def send_chunks(self, chunks: Iterable[WeightChunk]) -> None:
        """Send chunks via broadcast.

        Each chunk should contain exactly one parameter for broadcast strategy.
        Rank 0 sends the update request to inference engines, all ranks broadcast.

        Args:
            chunks: Iterable of WeightChunk objects to send.
        """
        for chunk in chunks:
            assert len(chunk) == 1, f"Broadcast strategy expects single-parameter chunks, got {len(chunk)}"

            name = chunk.names[0]
            tensor = chunk.tensors[0]
            shape = chunk.shapes[0]

            update_weight_task = None
            if self._rank == 0:
                # Create legacy update request and notify inference engines
                request: NamedWeightsUpdateRequest = {
                    "names": [name],
                    "dtypes": [self._model_dtype_str],
                    "shapes": [shape],
                }
                update_weight_task = asyncio.create_task(
                    self._inference_client.update_named_weights(request)
                )

            # Broadcast tensor from rank 0
            def broadcast_tensor(t: torch.Tensor) -> None:
                if self._rank == 0:
                    torch.distributed.broadcast(t.data, 0, group=self._model_update_group)

            await asyncio.to_thread(broadcast_tensor, tensor)

            if update_weight_task is not None:
                await update_weight_task

            torch.distributed.barrier()


class BroadcastWeightTransferReceiver(WeightTransferReceiver):
    """Receives weights via torch.distributed.broadcast.

    Allocates tensors locally and receives data via broadcast from training workers.
    """

    def __init__(
        self,
        model_dtype: torch.dtype,
        model_update_group: torch.distributed.ProcessGroup,
    ) -> None:
        """Initialize the broadcast receiver.

        Args:
            model_dtype: Expected dtype for received tensors.
            model_update_group: Process group for broadcast operations.
        """
        self._model_dtype = model_dtype
        self._model_update_group = model_update_group

    def receive_weights(
        self, request: NamedWeightsUpdateRequest
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via broadcast.

        Args:
            request: Weight update request with names, dtypes, shapes.

        Yields:
            Tuples of (parameter_name, tensor) for each weight.
        """
        from skyrl_train.utils import str_to_torch_dtype

        for name, dtype_str, shape in zip(request["names"], request["dtypes"], request["shapes"]):
            dtype = str_to_torch_dtype(dtype_str)
            assert dtype == self._model_dtype, f"dtype mismatch: request {dtype}, model {self._model_dtype}"

            # Allocate tensor and receive via broadcast
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)
            yield name, weight


class BroadcastTransferStrategy(WeightTransferStrategy):
    """Strategy for broadcast-based weight transfer using torch.distributed.

    This strategy uses NCCL/Gloo broadcast operations to transfer weights from
    training workers to inference engines. Requires a shared process group.
    """

    def __init__(
        self,
        rank: int,
        model_update_group: torch.distributed.ProcessGroup,
        model_dtype: torch.dtype,
    ) -> None:
        """Initialize the broadcast strategy.

        Args:
            rank: This process's rank in the distributed group.
            model_update_group: Process group for broadcast operations.
            model_dtype: Model dtype (e.g., torch.bfloat16).
        """
        from skyrl_train.utils import torch_dtype_to_str

        self._rank = rank
        self._model_update_group = model_update_group
        self._model_dtype = model_dtype
        self._dtype_str = torch_dtype_to_str(model_dtype)

    def create_sender(self) -> WeightTransferSender:
        """Create a broadcast sender."""
        return BroadcastWeightTransferSender(
            rank=self._rank,
            model_update_group=self._model_update_group,
            model_dtype_str=self._dtype_str,
        )

    def create_receiver(self) -> WeightTransferReceiver:
        """Create a broadcast receiver."""
        return BroadcastWeightTransferReceiver(
            model_dtype=self._model_dtype,
            model_update_group=self._model_update_group,
        )
