"""Ray Direct Transport (RDT) weight transfer strategy.

This module implements the RDT transfer strategy for synchronizing model weights
from training workers to inference engines using Ray's NCCL tensor transport.

RDT transfers GPU tensors between Ray actors via NCCL without requiring explicit
``torch.distributed`` process group setup. The sender packs weights into a
contiguous GPU buffer and passes it to inference engine actors using
``actor.method.options(tensor_transport="nccl").remote(tensor)``, which triggers
a GPU-to-GPU NCCL transfer managed entirely by Ray.

Key architectural differences from broadcast/IPC strategies:
- No ``torch.distributed`` process group needed -- Ray manages NCCL internally
- Actors must be created with ``enable_tensor_transport=True``
- Both sender and receiver actors must be in the same Ray collective group

Constraints for initial integration:
- Only supports Ray actor inference path (not HTTP server path)
- Only supports TP=PP=1 (single-GPU inference engines)
- Opt-in via ``weight_sync_backend: "rdt"``
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TYPE_CHECKING

import torch
from loguru import logger

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
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


class RdtWeightTransferSender(WeightTransferSender):
    """Sends weights via Ray Direct Transport (NCCL tensor transport).

    Packs all tensors in each chunk into a contiguous GPU buffer and sends
    the buffer plus metadata to inference engines via the inference client's
    ``update_weights_rdt`` method. The tensor is transferred GPU-to-GPU via
    Ray's NCCL tensor transport -- no CPU roundtrip or pickling.

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

        Packs tensors into a contiguous GPU buffer and sends the buffer plus
        metadata dict to inference engines via the inference client.

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
                metadata = {
                    "names": names,
                    "dtypes": dtypes,
                    "shapes": shapes,
                    "sizes": sizes,
                }
                await self._inference_client.update_weights_rdt(
                    packed_tensor=packed_tensor, metadata=metadata
                )

            torch.distributed.barrier()

    def teardown(self) -> None:
        """No-op for RDT sender (no custom process group to clean up)."""
        pass


class RdtWeightTransferReceiver(WeightTransferReceiver):
    """Receives weights via Ray Direct Transport.

    The packed tensor arrives already on GPU (transferred via NCCL).
    This receiver simply slices the packed tensor and yields
    ``(name, view)`` pairs.
    """

    def __init__(self, model_dtype: torch.dtype) -> None:
        self._model_dtype = model_dtype

    def receive_weights(self, packed_tensor: torch.Tensor, metadata: dict) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights from RDT transport.

        The packed tensor is already on GPU (transferred via NCCL by Ray).
        This method slices it into individual parameter tensors.

        Args:
            packed_tensor: Contiguous GPU tensor containing all packed weights.
            metadata: Dict with keys ``names``, ``dtypes``, ``shapes``, ``sizes``.

        Yields:
            Tuples of (parameter_name, tensor) for each weight.
        """
        from skyrl.train.utils.utils import str_to_torch_dtype

        if len(set(metadata["dtypes"])) != 1:
            raise ValueError("packed weight update should have all tensors with the same dtype")
        if str_to_torch_dtype(metadata["dtypes"][0]) != self._model_dtype:
            raise ValueError(
                f"mismatch dtype: src {metadata['dtypes'][0]}, dst {self._model_dtype}"
            )
        if len(metadata["sizes"]) != len(metadata["names"]):
            raise ValueError("sizes must have the same length as names")

        offset = 0
        for name, shape, size in zip(metadata["names"], metadata["shapes"], metadata["sizes"]):
            yield name, packed_tensor[offset : offset + size].view(*shape)
            offset += size

    def teardown(self) -> None:
        """No-op for RDT receiver (no custom resources to clean up)."""
        pass


class RdtTransferStrategy(WeightTransferStrategy):
    """Factory for RDT-based weight transfer.

    This strategy uses Ray's NCCL tensor transport to transfer weights from
    training workers to inference engines. Tensors stay on GPU throughout --
    no pickling or CPU copies.

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
