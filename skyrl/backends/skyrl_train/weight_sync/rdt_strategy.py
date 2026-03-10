"""Ray Direct Transport (RDT) weight transfer strategy.

This module implements the RDT transfer strategy for synchronizing model weights
from training workers to inference engines using Ray's built-in tensor transport.

RDT uses ``@ray.method(tensor_transport="nccl")`` decorated actor methods
to transfer GPU tensors between Ray actors via ObjectRef passing, without
requiring explicit ``torch.distributed`` process group setup.

Key architectural differences from broadcast/IPC strategies:
- Transfer is actor-level (ObjectRef), not process-level (torch.distributed)
- Orchestration happens outside actors (coordinator passes ObjectRef)
- No custom process group setup needed

Constraints for initial integration:
- Only supports Ray actor inference path (not HTTP server path)
- Only supports TP=PP=1 (single-GPU inference engines)
- Opt-in via ``weight_sync_backend: "rdt"``
"""

import asyncio
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TYPE_CHECKING

import torch
from loguru import logger

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk, WeightUpdateRequest
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferStrategy,
    WeightTransferSender,
    WeightTransferReceiver,
)

if TYPE_CHECKING:
    from skyrl.train.config.config import InferenceEngineConfig
    from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


@dataclass
class RdtInitInfo(WeightSyncInitInfo):
    """Initialization info for RDT-based weight transfer.

    RDT requires minimal init info since Ray handles the transport setup.
    """

    model_dtype_str: str

    @staticmethod
    def strategy_type() -> type:
        """Return the strategy class for this init info type."""
        return RdtTransferStrategy


@dataclass
class RdtWeightUpdateRequest(WeightUpdateRequest):
    """Request for RDT-based weight transfer.

    Contains metadata for unpacking the packed tensor received via RDT,
    plus the serialized packed tensor data.
    """

    sizes: List[int]  # Size in elements per parameter (for unpacking)
    packed_tensor_bytes: bytes  # Serialized packed tensor data

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize the request to JSON."""
        import base64

        data = {
            "names": self.names,
            "dtypes": self.dtypes,
            "shapes": self.shapes,
            "sizes": self.sizes,
            "packed_tensor_bytes": base64.b64encode(self.packed_tensor_bytes).decode("utf-8"),
        }
        return data

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "RdtWeightUpdateRequest":
        """Deserialize the request from JSON."""
        import base64

        data = data.copy()
        data["packed_tensor_bytes"] = base64.b64decode(data["packed_tensor_bytes"])
        return cls(**data)


class RdtWeightTransferSender(WeightTransferSender):
    """Sends weights via Ray Direct Transport.

    Packs all tensors in each chunk into a contiguous buffer, serializes it,
    and sends via the inference client. The receiver unpacks the buffer.

    Only rank 0 sends requests to inference engines. Other training ranks
    participate in weight extraction (FSDP collective) but don't send.
    """

    def __init__(
        self,
        init_info: RdtInitInfo,
        inference_client: "InferenceEngineClient",
    ) -> None:
        self._init_info = init_info
        self._inference_client = inference_client

    async def send_chunks(self, chunks: Iterable[WeightChunk]) -> None:
        """Send chunks via RDT.

        Packs tensors into a contiguous buffer, serializes to bytes, and
        sends to inference engines via the inference client.

        All training ranks iterate through chunks (weight extraction may
        involve collective ops), but only rank 0 sends to inference engines.

        Args:
            chunks: Iterable of WeightChunk objects to send.
        """
        from skyrl.train.utils.utils import str_to_torch_dtype

        rank = torch.distributed.get_rank()
        device = torch.cuda.current_device()
        dtype = str_to_torch_dtype(self._init_info.model_dtype_str)

        for chunk in chunks:
            names = []
            dtypes = []
            shapes = []
            sizes = []

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
                names.append(name)
                dtypes.append(self._init_info.model_dtype_str)
                shapes.append(shape)
                sizes.append(size)

            torch.distributed.barrier()
            torch.cuda.synchronize()

            # Only rank 0 sends to inference engines
            if rank == 0:
                # Serialize the packed tensor to bytes for transport
                packed_tensor_bytes = pickle.dumps(packed_tensor.cpu())

                request = RdtWeightUpdateRequest(
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    sizes=sizes,
                    packed_tensor_bytes=packed_tensor_bytes,
                )
                await self._inference_client.update_named_weights(request)

            torch.distributed.barrier()
            torch.cuda.synchronize()

    def teardown(self) -> None:
        """No-op for RDT sender (no custom process group to clean up)."""
        pass


class RdtWeightTransferReceiver(WeightTransferReceiver):
    """Receives weights via Ray Direct Transport.

    Deserializes the packed tensor from bytes and unpacks individual
    parameter tensors.
    """

    def __init__(self, model_dtype: torch.dtype) -> None:
        self._model_dtype = model_dtype

    def receive_weights(self, request: RdtWeightUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights from RDT transport.

        Deserializes the packed tensor from bytes, moves to GPU, and
        unpacks individual parameter tensors.

        Args:
            request: RDT weight update request with packed tensor bytes.

        Yields:
            Tuples of (parameter_name, tensor) for each weight.
        """
        from skyrl.train.utils.utils import str_to_torch_dtype

        assert len(set(request.dtypes)) == 1, "packed weight update should have all tensors with the same dtype"
        assert (
            str_to_torch_dtype(request.dtypes[0]) == self._model_dtype
        ), f"mismatch dtype: src {request.dtypes[0]}, dst {self._model_dtype}"
        assert len(request.sizes) == len(request), "sizes must be provided for packed weight update"

        # Deserialize packed tensor and move to GPU
        packed_tensor_cpu = pickle.loads(request.packed_tensor_bytes)
        packed_tensor = packed_tensor_cpu.to(device=f"cuda:{torch.cuda.current_device()}", dtype=self._model_dtype)

        offset = 0
        for name, shape, size in zip(request.names, request.shapes, request.sizes):
            yield name, packed_tensor[offset : offset + size].view(*shape)
            offset += size

    def teardown(self) -> None:
        """No-op for RDT receiver (no custom resources to clean up)."""
        pass


class RdtTransferStrategy(WeightTransferStrategy):
    """Factory for RDT-based weight transfer.

    This strategy uses Ray's tensor transport to transfer weights from
    training workers to inference engines. It packs tensors and sends
    them as serialized bytes through the existing inference client
    update_named_weights path.

    Constraints:
    - Only supports Ray actor inference path (not HTTP server path)
    - Only supports TP=PP=1 (single-GPU inference engines)

    All methods are static - no instance state needed.
    """

    @staticmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig", inference_world_size: Optional[int] = None
    ) -> RdtInitInfo:
        """Create init info with all config-derived args."""
        if ie_cfg.tensor_parallel_size > 1 or ie_cfg.pipeline_parallel_size > 1:
            raise ValueError(
                "RDT weight sync backend currently only supports TP=PP=1 "
                f"(got TP={ie_cfg.tensor_parallel_size}, PP={ie_cfg.pipeline_parallel_size}). "
                "Use 'nccl' backend for multi-GPU inference engines."
            )

        return RdtInitInfo(
            model_dtype_str=ie_cfg.model_dtype,
            override_existing_receiver=ie_cfg.override_existing_update_group == "enable",
        )

    @staticmethod
    def create_sender(
        init_info: RdtInitInfo,
        inference_client: "InferenceEngineClient",
    ) -> RdtWeightTransferSender:
        """Create an RDT sender.

        Args:
            init_info: RdtInitInfo containing config-derived args.
            inference_client: Client for coordinating with inference engines.

        Returns:
            A configured RdtWeightTransferSender instance.
        """
        return RdtWeightTransferSender(
            init_info=init_info,
            inference_client=inference_client,
        )

    @staticmethod
    def create_receiver(init_info: RdtInitInfo) -> RdtWeightTransferReceiver:
        """Create an RDT receiver.

        Args:
            init_info: RdtInitInfo from the sender.

        Returns:
            A configured RdtWeightTransferReceiver instance.
        """
        from skyrl.train.utils.utils import str_to_torch_dtype

        model_dtype = str_to_torch_dtype(init_info.model_dtype_str)
        return RdtWeightTransferReceiver(model_dtype=model_dtype)
