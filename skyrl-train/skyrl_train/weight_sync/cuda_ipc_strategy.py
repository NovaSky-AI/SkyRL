"""CUDA IPC-based weight transfer strategy.

This module implements the CUDA IPC transfer strategy for synchronizing model weights
from training workers to inference engines using CUDA IPC handles.
"""

import asyncio
from typing import Iterable, Iterator, Tuple

import torch
from torch.multiprocessing.reductions import reduce_tensor

from skyrl_train.inference_engines.base import NamedWeightsUpdateRequest
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.utils import get_physical_gpu_id, str_to_torch_dtype
from skyrl_train.weight_sync.base import WeightChunk
from skyrl_train.weight_sync.transfer_strategy import (
    WeightTransferStrategy,
    WeightTransferSender,
    WeightTransferReceiver,
)


class CudaIpcWeightTransferSender(WeightTransferSender):
    """Sends weights via CUDA IPC handles.

    Creates IPC handles for tensors, gathers them across ranks, and sends
    the handle metadata to inference engines.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        model_dtype_str: str,
        inference_client: InferenceEngineClient,
    ) -> None:
        """Initialize the CUDA IPC sender.

        Args:
            rank: This worker's rank in the distributed group.
            world_size: Total number of workers in the distributed group.
            model_dtype_str: Model dtype as string (e.g., "bfloat16") for request metadata.
            inference_client: Client for coordinating with inference engines.
        """
        self._rank = rank
        self._world_size = world_size
        self._model_dtype_str = model_dtype_str
        self._inference_client = inference_client

    async def send_chunks(self, chunks: Iterable[WeightChunk]) -> None:
        """Send chunks via CUDA IPC.

        Each chunk can contain multiple parameters (batched). IPC handles are
        created for each tensor and gathered across all ranks.

        Args:
            chunks: Iterable of WeightChunk objects to send.
        """
        for chunk in chunks:
            weights_update_request: NamedWeightsUpdateRequest = {
                "names": [],
                "dtypes": [],
                "shapes": [],
                "extras": [],
                "packed": False,
            }

            # Process all parameters in this batch
            # TODO(haochen): Pack tensors into contiguous buffer before creating IPC handle
            # (like Megatron does) to reduce number of IPC handles and file descriptors
            for name, tensor, shape in zip(chunk.names, chunk.tensors, chunk.shapes):
                # Create IPC handle for tensor
                ipc_handle = reduce_tensor(tensor)
                ipc_handle = {get_physical_gpu_id(): ipc_handle}
                ipc_handle_list = [None] * self._world_size
                torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                if self._rank == 0:
                    ipc_handles = {}
                    for d in ipc_handle_list:
                        ipc_handles.update(d)

                    weights_update_request["names"].append(name)
                    weights_update_request["dtypes"].append(self._model_dtype_str)
                    weights_update_request["shapes"].append(shape)
                    weights_update_request["extras"].append({"ipc_handles": ipc_handles})

                torch.distributed.barrier()
                torch.cuda.synchronize()

            # Send batch
            if self._rank == 0:
                await self._inference_client.update_named_weights(weights_update_request)
                torch.cuda.ipc_collect()
            torch.distributed.barrier()
            torch.cuda.synchronize()


class CudaIpcWeightTransferReceiver(WeightTransferReceiver):
    """Receives weights via CUDA IPC handles.

    Opens IPC handles to access tensors shared from training workers.
    """

    def __init__(
        self,
        model_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Initialize the CUDA IPC receiver.

        Args:
            model_dtype: Expected dtype for received tensors.
            device: CUDA device for this worker.
        """
        self._model_dtype = model_dtype
        self._device = device

    def receive_weights(
        self, request: NamedWeightsUpdateRequest
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via CUDA IPC handles.

        Args:
            request: Weight update request with names, dtypes, shapes, and IPC handles.

        Yields:
            Tuples of (parameter_name, tensor) for each weight.
        """
        names = request["names"]
        dtypes = request["dtypes"]
        shapes = request["shapes"]
        sizes = request.get("sizes", [])
        ipc_handles = [extra["ipc_handles"] for extra in request["extras"]]
        packed = request.get("packed", False)

        if packed:
            yield from self._receive_packed(names, dtypes, shapes, sizes, ipc_handles)
        else:
            yield from self._receive_unpacked(names, dtypes, shapes, ipc_handles)

    def _receive_packed(
        self,
        names: list,
        dtypes: list,
        shapes: list,
        sizes: list,
        ipc_handles: list,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive packed tensors from a single IPC handle."""
        assert len(ipc_handles) == 1, "packed weight update should receive one ipc handle for all tensors"
        assert len(set(dtypes)) == 1, "packed weight update should have all tensors with the same dtype"
        assert (
            str_to_torch_dtype(dtypes[0]) == self._model_dtype
        ), f"mismatch dtype: src {dtypes[0]}, dst {self._model_dtype}"
        assert len(sizes) == len(names), "sizes must be provided for packed weight update"
        assert all(isinstance(size, int) for size in sizes), "sizes should be a list of integers"

        cuda_device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(cuda_device)
        physical_gpu_id = str(props.uuid)

        handle = ipc_handles[0][physical_gpu_id]
        device_id = self._device.index
        func, args = handle
        list_args = list(args)
        list_args[6] = device_id
        packed_tensor = func(*list_args)

        offset = 0
        for name, shape, size in zip(names, shapes, sizes):
            yield name, packed_tensor[offset : offset + size].view(*shape)
            offset += size

    def _receive_unpacked(
        self,
        names: list,
        dtypes: list,
        shapes: list,
        ipc_handles: list,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive individual tensors from separate IPC handles."""
        cuda_device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(cuda_device)
        physical_gpu_id = str(props.uuid)

        for name, dtype_str, shape, ipc_handle in zip(names, dtypes, shapes, ipc_handles):
            dtype = str_to_torch_dtype(dtype_str)
            assert dtype == self._model_dtype, f"mismatch dtype: src {dtype}, dst {self._model_dtype}"

            handle = ipc_handle[physical_gpu_id]
            device_id = self._device.index
            func, args = handle
            list_args = list(args)
            list_args[6] = device_id
            weight = func(*list_args)
            yield name, weight


class CudaIpcTransferStrategy(WeightTransferStrategy):
    """Strategy for CUDA IPC-based weight transfer.

    This strategy uses CUDA IPC handles to share GPU memory between training
    workers and inference engines on the same machine.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        model_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Initialize the CUDA IPC strategy.

        Args:
            rank: This process's rank in the distributed group.
            world_size: Total number of workers in the distributed group.
            model_dtype: Model dtype (e.g., torch.bfloat16).
            device: CUDA device for this worker.
        """
        from skyrl_train.utils import torch_dtype_to_str

        self._rank = rank
        self._world_size = world_size
        self._model_dtype = model_dtype
        self._model_dtype_str = torch_dtype_to_str(model_dtype)
        self._device = device

    def create_sender(self) -> WeightTransferSender:
        """Create a CUDA IPC sender."""
        return CudaIpcWeightTransferSender(
            rank=self._rank,
            world_size=self._world_size,
            model_dtype_str=self._model_dtype_str,
        )

    def create_receiver(self) -> WeightTransferReceiver:
        """Create a CUDA IPC receiver."""
        return CudaIpcWeightTransferReceiver(
            model_dtype=self._model_dtype,
            device=self._device,
        )
