"""CUDA IPC-based weight transfer strategy.

This module implements the CUDA IPC transfer strategy for synchronizing model weights
from training workers to inference engines using CUDA IPC handles.
"""

import asyncio
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
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.config import InferenceEngineConfig

import torch

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk, WeightUpdateRequest
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferSender,
    WeightTransferStrategy,
)

# IPC handle type: (rebuild_func, args) returned by reduce_tensor
IpcHandle = Tuple[Callable[..., torch.Tensor], Tuple[Any, ...]]


@dataclass
class CudaIpcInitInfo(WeightSyncInitInfo):
    """Initialization info for CUDA IPC-based weight transfer."""

    model_dtype_str: str

    def for_servers(self, world_size_per_server: int, num_servers: int, dp_size: int = 1) -> List["CudaIpcInitInfo"]:
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
        inference_client: "RemoteInferenceClient",
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
        await self._send_chunks_vllm_native(chunks, weight_metadata)

    async def _send_chunks_vllm_native(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
    ) -> None:
        """Send weights via vLLM's native CUDA IPC weight transfer.

        vLLM 0.23.0 receives weights natively: the inference worker's
        ``weight_transfer_engine`` (``weight_transfer_config.backend="ipc"``)
        unpacks the packed CUDA-IPC payload via ``packed_ipc_consumer`` and
        loads it. We delegate the trainer side to vLLM's own
        ``IPCWeightTransferEngine.trainer_send_weights`` (packed mode), which
        handles uint8 packing, the cross-rank IPC-handle all-gather, bounded
        per-chunk buffers, and barriers. We drive the lifecycle over the
        inference client: ``/start_weight_update`` -> ``/update_weights``
        (invoked by the ``send_mode`` callback, once per packed chunk) ->
        ``/finish_weight_update``.

        ``trainer_send_weights`` is synchronous; we run it in a worker thread
        and bridge its rank-0 ``send_mode`` callback back to this event loop so
        the native ``IPCWeightTransferUpdateInfo`` payload is fanned out to all
        inference servers via the (async) client.
        """
        from vllm.distributed.weight_transfer.ipc_engine import (
            IPCTrainerSendWeightsArgs,
            IPCWeightTransferEngine,
        )

        loop = asyncio.get_running_loop()
        rank = torch.distributed.get_rank()

        def weight_iterator() -> Iterator[Tuple[str, torch.Tensor]]:
            for chunk in chunks:
                yield from zip(chunk.names, chunk.tensors)

        def send_update(update_info: Any) -> None:
            # Called by trainer_send_weights on rank 0 (inside the worker
            # thread). Build the HTTP-friendly native update payload (IPC
            # handles base64-pickled, mirroring vLLM's own 'http' send_mode)
            # and fan out to all servers via the async client, blocking until
            # done. We read fields off update_info directly rather than
            # asdict() to avoid deep-copying the IPC handle args.
            fields = {
                "update_kind": update_info.update_kind,
                "names": update_info.names,
                "dtype_names": update_info.dtype_names,
                "shapes": update_info.shapes,
                "tensor_sizes": update_info.tensor_sizes,
                "packed": update_info.packed,
                "ipc_handles_pickled": base64.b64encode(pickle.dumps(update_info.ipc_handles)).decode("utf-8"),
            }
            future = asyncio.run_coroutine_threadsafe(self._inference_client.update_named_weights(fields), loop)
            future.result()

        if rank == 0:
            await self._inference_client.start_weight_update(is_checkpoint_format=True)
        torch.distributed.barrier()

        # Blocking: packs each chunk into a uint8 buffer, all-gathers IPC
        # handles across ranks, and (rank 0) invokes send_update per chunk.
        await asyncio.to_thread(
            IPCWeightTransferEngine.trainer_send_weights,
            iterator=weight_iterator(),
            trainer_args=IPCTrainerSendWeightsArgs(send_mode=send_update, packed=True),
        )

        if rank == 0:
            await self._inference_client.finish_weight_update()
        torch.distributed.barrier()

    def teardown(self) -> None:
        """No-op for CUDA IPC sender (no custom process group to clean up)."""
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
            override_existing_receiver=not ie_cfg.run_engines_locally,
        )

    @staticmethod
    def create_sender(
        init_info: CudaIpcInitInfo,
        inference_client: "RemoteInferenceClient",
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
    def get_vllm_transfer_engine() -> type:
        """Return the vLLM weight-transfer engine class for this strategy (CUDA IPC).

        Reference for the receive side: the inference servers drive this engine
        natively. Currently unused on the sender side (we route through the
        SkyRL ``/collective_rpc`` wrap), kept as the canonical mapping.
        """
        from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

        return IPCWeightTransferEngine
