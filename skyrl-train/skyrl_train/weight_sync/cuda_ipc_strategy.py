"""CUDA IPC-based weight transfer strategy.

This module implements the CUDA IPC transfer strategy for synchronizing model weights
from training workers to inference engines using CUDA IPC handles.
"""

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import torch
from torch.multiprocessing.reductions import reduce_tensor

from skyrl_train.inference_engines.base import NamedWeightsUpdateRequest
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.utils import get_physical_gpu_id, str_to_torch_dtype
from skyrl_train.weight_sync.base import WeightChunk
from skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferStrategy,
    WeightTransferSender,
    WeightTransferReceiver,
)


@dataclass
class CudaIpcInitInfo(WeightSyncInitInfo):
    """Initialization info for CUDA IPC-based weight transfer."""
    model_dtype_str: str

    @staticmethod
    def strategy_type() -> type:
        """Return the strategy class for this init info type."""
        return CudaIpcTransferStrategy


class CudaIpcWeightTransferSender(WeightTransferSender):
    """Sends weights via CUDA IPC handles.

    Creates IPC handles for tensors, gathers them across ranks, and sends
    the handle metadata to inference engines.
    """

    def __init__(
        self,
        init_info: CudaIpcInitInfo,
        inference_client: InferenceEngineClient,
    ) -> None:
        """Initialize the CUDA IPC sender.

        Args:
            init_info: CudaIpcInitInfo containing config-derived args.
            inference_client: Client for coordinating with inference engines.
        """
        self._init_info = init_info
        self._inference_client = inference_client

    async def send_chunks(self, chunks: Iterable[WeightChunk]) -> None:
        """Send chunks via CUDA IPC with packed tensors.

        Each chunk can contain multiple parameters. All tensors in a chunk are
        packed into a single contiguous buffer, and one IPC handle is created
        for the packed buffer. This reduces the number of IPC handles and file
        descriptors.

        Args:
            chunks: Iterable of WeightChunk objects to send.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = torch.cuda.current_device()
        dtype = str_to_torch_dtype(self._init_info.model_dtype_str)

        for chunk in chunks:
            weights_update_request: NamedWeightsUpdateRequest = {
                "names": [],
                "dtypes": [],
                "shapes": [],
                "sizes": [],
                "extras": [],
                "packed": True,
            }

            # Pack all tensors in this chunk into a single contiguous buffer
            total_numel = sum(t.numel() for t in chunk.tensors)
            packed_tensor = torch.empty(
                total_numel,
                device=device,
                dtype=dtype,
                requires_grad=False,
            )

            offset = 0
            for name, tensor, shape in zip(chunk.names, chunk.tensors, chunk.shapes):
                size = tensor.numel()
                packed_tensor[offset : offset + size].copy_(tensor.detach().view(-1))
                offset += size
                weights_update_request["names"].append(name)
                weights_update_request["dtypes"].append(self._init_info.model_dtype_str)
                weights_update_request["shapes"].append(shape)
                weights_update_request["sizes"].append(size)

            # Create single IPC handle for the packed buffer
            ipc_handle = reduce_tensor(packed_tensor)
            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * world_size
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if rank == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)
                weights_update_request["extras"].append({"ipc_handles": ipc_handles})

            torch.distributed.barrier()
            torch.cuda.synchronize()

            # Send the packed chunk
            if rank == 0:
                await self._inference_client.update_named_weights(weights_update_request)

            torch.cuda.ipc_collect()
            torch.distributed.barrier()
            torch.cuda.synchronize()


class CudaIpcWeightTransferReceiver(WeightTransferReceiver):
    """Receives weights via CUDA IPC handles.

    Opens IPC handles to access tensors shared from training workers.
    """

    def __init__(self, model_dtype: torch.dtype) -> None:
        """Initialize the CUDA IPC receiver.

        Args:
            model_dtype: Expected dtype for received tensors.
        """
        self._model_dtype = model_dtype

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

        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)
        physical_gpu_id = str(props.uuid)

        handle = ipc_handles[0][physical_gpu_id]
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
        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)
        physical_gpu_id = str(props.uuid)

        for name, dtype_str, shape, ipc_handle in zip(names, dtypes, shapes, ipc_handles):
            dtype = str_to_torch_dtype(dtype_str)
            assert dtype == self._model_dtype, f"mismatch dtype: src {dtype}, dst {self._model_dtype}"

            handle = ipc_handle[physical_gpu_id]
            func, args = handle
            list_args = list(args)
            list_args[6] = device_id
            weight = func(*list_args)
            yield name, weight


class CudaIpcTransferStrategy(WeightTransferStrategy):
    """Factory for CUDA IPC-based weight transfer.

    This strategy uses CUDA IPC handles to share GPU memory between training
    workers and inference engines on the same machine.

    All methods are static - no instance state needed.
    """

    @staticmethod
    def create_init_info(cfg: "DictConfig") -> CudaIpcInitInfo:
        """Create init info with all config-derived args.

        Args:
            cfg: Configuration object containing generator settings.

        Returns:
            CudaIpcInitInfo containing all args needed for sender/receiver creation.
        """
        return CudaIpcInitInfo(model_dtype_str=cfg.generator.model_dtype)

    @staticmethod
    def create_sender(
        init_info: CudaIpcInitInfo,
        inference_client: InferenceEngineClient,
    ) -> CudaIpcWeightTransferSender:
        """Create a CUDA IPC sender.

        Args:
            init_info: CudaIpcInitInfo containing config-derived args.
            inference_client: Client for coordinating with inference engines.

        Returns:
            A configured CudaIpcWeightTransferSender instance.
        """
        return CudaIpcWeightTransferSender(
            init_info=init_info,
            inference_client=inference_client,
        )

    @staticmethod
    def create_receiver(init_info: CudaIpcInitInfo) -> CudaIpcWeightTransferReceiver:
        """Create a CUDA IPC receiver.

        Args:
            init_info: CudaIpcInitInfo from the sender.

        Returns:
            A configured CudaIpcWeightTransferReceiver instance.
        """
        from skyrl_train.utils import str_to_torch_dtype

        model_dtype = str_to_torch_dtype(init_info.model_dtype_str)
        return CudaIpcWeightTransferReceiver(model_dtype=model_dtype)
