"""CUDA IPC-based weight transfer strategy.

This module implements the CUDA IPC transfer strategy for synchronizing model weights
from training workers to inference engines using CUDA IPC handles.
"""

import base64
import copy
import pickle
from dataclasses import asdict, dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    from skyrl.train.config import InferenceEngineConfig

import torch
from torch.multiprocessing.reductions import reduce_tensor

from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk, WeightUpdateRequest
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferReceiver,
    WeightTransferSender,
    WeightTransferStrategy,
)
from skyrl.env_vars import _SKYRL_USE_NEW_INFERENCE
from skyrl.train.utils.utils import get_physical_gpu_id, str_to_torch_dtype

# IPC handle type: (rebuild_func, args) returned by reduce_tensor
IpcHandle = Tuple[Callable[..., torch.Tensor], Tuple[Any, ...]]


@dataclass
class CudaIpcInitInfo(WeightSyncInitInfo):
    """Initialization info for CUDA IPC-based weight transfer."""

    model_dtype_str: str

    @staticmethod
    def strategy_type() -> type:
        """Return the strategy class for this init info type."""
        return CudaIpcTransferStrategy

    def for_servers(self, world_size_per_server: int, num_servers: int) -> List["CudaIpcInitInfo"]:
        """IPC init is a no-op, so return identical copies for each server."""
        return [copy.deepcopy(self) for _ in range(num_servers)]

    def to_api_payload(self) -> Dict[str, Any]:
        """IPC needs no initialization parameters."""
        return {}


_IPC_REQUEST_END_MARKER = b"__END_OF_REQUEST__"


@dataclass
class CudaIpcWeightUpdateRequest(WeightUpdateRequest):
    """Request for CUDA IPC-based weight transfer.

    Contains IPC handles for direct GPU memory access. Tensors are packed into
    a contiguous buffer to reduce the number of IPC handles.
    """

    sizes: List[int]  # Size in elements per parameter (for unpacking)
    ipc_handles: Dict[str, IpcHandle]  # Physical GPU UUID -> IPC handle for the packed buffer

    def serialize(self) -> bytes:
        """Serialize the request to bytes."""
        import base64
        import pickle

        request_data = pickle.dumps(self)
        request_data_encoded = base64.b64encode(request_data)
        data_with_marker = request_data_encoded + _IPC_REQUEST_END_MARKER

        # Pad for 4-byte alignment
        data_size = len(data_with_marker)
        padded_size = ((data_size + 3) // 4) * 4
        result = bytearray(data_with_marker)
        result.extend(b"\x00" * (padded_size - data_size))
        return bytes(result)

    @classmethod
    def deserialize(cls, data: bytes) -> "CudaIpcWeightUpdateRequest":
        """Deserialize the request from bytes."""
        import base64
        import pickle

        end_index = data.find(_IPC_REQUEST_END_MARKER)
        if end_index == -1:
            raise ValueError("End marker not found in serialized data")
        request_data = data[:end_index]
        try:
            request_data_decoded = base64.b64decode(request_data)
            return pickle.loads(request_data_decoded)
        except Exception as e:
            raise ValueError("Failed to deserialize request") from e

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize the request to JSON."""
        data = asdict(self)
        # serialize the ipc handle
        import base64
        import pickle

        data["ipc_handles"] = base64.b64encode(pickle.dumps(self.ipc_handles)).decode("utf-8")
        return data

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "CudaIpcWeightUpdateRequest":
        """Deserialize the request from JSON."""
        import base64
        import pickle

        data = data.copy()
        data["ipc_handles"] = pickle.loads(base64.b64decode(data["ipc_handles"]))
        return cls(**data)


class CudaIpcWeightTransferSender(WeightTransferSender):
    """Sends weights via CUDA IPC handles.

    Creates IPC handles for tensors, gathers them across ranks, and sends
    the handle metadata to inference engines. When using the new inference
    path, sends handles via vLLM's native /update_weights endpoint.
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

    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
    ) -> None:
        """Send chunks via CUDA IPC.

        Args:
            chunks: Iterable of WeightChunk objects to send.
            weight_metadata: Unused for IPC (metadata is derived from chunks
                directly to avoid ordering mismatches). Kept for interface
                compatibility with the base class.
        """
        if _SKYRL_USE_NEW_INFERENCE:
            await self._send_chunks_vllm_native(chunks, weight_metadata)
        else:
            await self._send_chunks_legacy(chunks)

    async def _send_chunks_vllm_native(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
    ) -> None:
        """Send all weights via vLLM native IPC (new inference path).

        All ranks iterate chunks (weight extraction may use collective ops),
        create per-tensor IPC handles, and all-gather them so rank 0 has
        handles for every GPU. Rank 0 then sends the merged handles to
        vLLM via /update_weights with pickled IPC handles.

        Names, dtypes, and shapes are derived from the chunks themselves
        (not from weight_metadata) to guarantee they stay aligned with the
        IPC handles. FSDP state_dict() can return parameters in different
        orders across separate calls.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

        names: List[str] = []
        dtype_names: List[str] = []
        shapes: List[List[int]] = []
        tensor_refs: List[torch.Tensor] = []
        per_param_handles: List[Dict[str, IpcHandle]] = []
        for chunk in chunks:
            for name, tensor in zip(chunk.names, chunk.tensors):
                weight = tensor.detach().contiguous()
                tensor_refs.append(weight)
                handle = reduce_tensor(weight)
                per_param_handles.append({gpu_uuid: handle})
                names.append(name)
                dtype_names.append(str(weight.dtype).split(".")[-1])
                shapes.append(list(weight.shape))

        # All-gather handles from every training rank
        gathered: List[Optional[List[Dict[str, IpcHandle]]]] = [None] * world_size
        torch.distributed.all_gather_object(gathered, per_param_handles)

        torch.distributed.barrier()
        torch.cuda.synchronize()

        if rank == 0:
            # Merge per-parameter handle dicts across ranks so each
            # parameter's dict maps every GPU UUID to its handle.
            merged_handles: List[Dict[str, IpcHandle]] = []
            for param_idx in range(len(per_param_handles)):
                merged: Dict[str, IpcHandle] = {}
                for rank_handles in gathered:
                    merged.update(rank_handles[param_idx])  # type: ignore[union-attr]
                merged_handles.append(merged)

            pickled = base64.b64encode(pickle.dumps(merged_handles)).decode("utf-8")
            update_info: Dict[str, Any] = {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "ipc_handles_pickled": pickled,
            }
            await self._inference_client.update_named_weights(update_info)

        torch.distributed.barrier()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    async def _send_chunks_legacy(self, chunks: Iterable[WeightChunk]) -> None:
        """Per-chunk CUDA IPC with packed tensors (legacy path)."""
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = torch.cuda.current_device()
        dtype = str_to_torch_dtype(self._init_info.model_dtype_str)

        for chunk in chunks:
            names = []
            dtypes = []
            shapes = []
            sizes = []

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
                packed_tensor[offset : offset + size].copy_(tensor.detach().reshape(-1))
                offset += size
                names.append(name)
                dtypes.append(self._init_info.model_dtype_str)
                shapes.append(shape)
                sizes.append(size)

            ipc_handle: IpcHandle = reduce_tensor(packed_tensor)
            ipc_handle_dict: Dict[str, IpcHandle] = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list: List[Optional[Dict[str, IpcHandle]]] = [None] * world_size
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle_dict)

            ipc_handles: Dict[str, IpcHandle] = {}
            if rank == 0:
                for d in ipc_handle_list:
                    if d is not None:
                        ipc_handles.update(d)

            torch.distributed.barrier()
            torch.cuda.synchronize()

            if rank == 0:
                request = CudaIpcWeightUpdateRequest(
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    sizes=sizes,
                    ipc_handles=ipc_handles,
                )
                await self._inference_client.update_named_weights(request)

            torch.cuda.ipc_collect()
            torch.distributed.barrier()
            torch.cuda.synchronize()

    def teardown(self) -> None:
        """No-op for CUDA IPC sender (no custom process group to clean up)."""
        pass


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

    def receive_weights(self, request: CudaIpcWeightUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via CUDA IPC handles.

        Args:
            request: CUDA IPC weight update request with names, dtypes, shapes, sizes, and IPC handles.

        Yields:
            Tuples of (parameter_name, tensor) for each weight.
        """
        assert len(set(request.dtypes)) == 1, "packed weight update should have all tensors with the same dtype"
        assert (
            str_to_torch_dtype(request.dtypes[0]) == self._model_dtype
        ), f"mismatch dtype: src {request.dtypes[0]}, dst {self._model_dtype}"
        assert len(request.sizes) == len(request), "sizes must be provided for packed weight update"
        assert all(isinstance(size, int) for size in request.sizes), "sizes should be a list of integers"

        device_id = torch.cuda.current_device()
        physical_gpu_id = get_physical_gpu_id()

        handle = request.ipc_handles[physical_gpu_id]
        func, args = handle
        list_args = list(args)
        list_args[6] = device_id
        packed_tensor = func(*list_args)

        offset = 0
        for name, shape, size in zip(request.names, request.shapes, request.sizes):
            yield name, packed_tensor[offset : offset + size].view(*shape)
            offset += size

    def teardown(self) -> None:
        """No-op for CUDA IPC receiver (no custom process group to clean up)."""
        pass


class CudaIpcTransferStrategy(WeightTransferStrategy):
    """Factory for CUDA IPC-based weight transfer.

    This strategy uses CUDA IPC handles to share GPU memory between training
    workers and inference engines on the same machine.

    All methods are static - no instance state needed.
    """

    @staticmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig", inference_world_size: Optional[int] = None
    ) -> CudaIpcInitInfo:
        """Create init info with all config-derived args."""
        return CudaIpcInitInfo(
            model_dtype_str=ie_cfg.model_dtype,
            override_existing_receiver=ie_cfg.override_existing_update_group == "enable",
        )

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
        from skyrl.train.utils.utils import str_to_torch_dtype

        model_dtype = str_to_torch_dtype(init_info.model_dtype_str)
        return CudaIpcWeightTransferReceiver(model_dtype=model_dtype)
