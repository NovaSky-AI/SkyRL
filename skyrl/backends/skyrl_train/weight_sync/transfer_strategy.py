"""Weight transfer strategy abstractions for distributed RL training.

This module defines the abstract interfaces for transferring model weights
from training workers to inference engines. The strategy pattern allows different
transfer mechanisms (broadcast, CUDA IPC) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterable, Optional

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk

if TYPE_CHECKING:
    import torch

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.config import InferenceEngineConfig


@dataclass
class WeightSyncInitInfo(ABC):
    """Base class for weight sync initialization info."""

    override_existing_receiver: bool
    """Whether to override an existing weight receiver. If False and a receiver exists, init is skipped."""


class WeightTransferSender(ABC):
    """Strategy-specific component that sends WeightChunk data to inference actors.

    Implementations handle the transfer primitive (broadcast, CUDA IPC) and coordinate
    with inference actors.
    """

    handles_prefix_cache_reset: bool = False
    """Indicates whether the transfer strategy handles resetting prefix cache
        for the inference engines internally."""

    force_disable_expandable_segments: ClassVar[bool] = False
    """Disable expandable_segments around the send even when NOT colocated.

    The push backends only need it under ``colocate_all`` (CUDA IPC calls
    cudaIpcGetMemHandle, which VMM addresses break). A backend that shares GPU
    memory on every run regardless of colocation sets this True."""

    empty_cache_after_send: ClassVar[bool] = True
    """Whether the worker should ``torch.cuda.empty_cache()`` after the send.

    False for backends whose send buffers are reused by the next step, where
    scrubbing them back to CUDA is pure cost. A colocated inference engine needs
    the physical memory regardless, so the worker still empties under
    ``colocate_all``."""

    async def send(
        self,
        weight_extractor: Any,
        dtype: "torch.dtype",
        **kwargs,
    ) -> None:
        """Send this rank's weights. Called on every training rank.

        The default materializes the extractor's chunk stream plus its metadata
        and hands both to :meth:`send_chunks` — the push backends' contract.
        Backends that do not consume a chunk stream override this instead, which
        is what keeps ``get_weight_metadata`` (a whole-model gather on the
        Megatron extractor) off their critical path entirely.

        Args:
            weight_extractor: The worker's extractor, already built.
            dtype: Inference dtype to convert to.
            **kwargs: Forwarded to :meth:`send_chunks`.
        """
        await self.send_chunks(
            weight_extractor.extract_weights(dtype),
            weight_metadata=weight_extractor.get_weight_metadata(dtype),
            **kwargs,
        )

    @abstractmethod
    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
        **kwargs,
    ) -> None:
        """Send chunks using this transfer strategy.

        This method must be called on all training ranks. Implementations may have
        different behavior for different ranks.

        Args:
            chunks: Iterable of WeightChunk objects to send.
            weight_metadata: Optional pre-computed metadata (names, dtype_names, shapes).
                When provided, allows the sender to avoid materializing all chunks
                to collect metadata upfront.
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources used by the sender (e.g., destroy process groups)."""
        ...


# NOTE (sumanthrh): WeightTransferStrategy is assymetric - only dictates sender send APIs
# because we rely on the native vLLM WeightTransferEngine for the receive logic.
# For CUDA IPC, we use a custom send implementation and for NCCL, we rely on
# vLLM's NCCLWeightTransferEngine for the send logic.
class WeightTransferStrategy(ABC):
    """Stateless factory for creating init info and senders.

    Each strategy implementation provides static methods to create:
    - init_info: Contains all config-derived args
    - sender: Uses init_info + inference_client

    Usage on sender side:
        init_info = Strategy.create_init_info(ie_cfg, inference_world_size)
        sender = Strategy.create_sender(init_info, inference_client)

    The receiver side lives inside the inference servers (vLLM's native weight
    transfer engine), driven via the inference client's HTTP control plane.
    """

    sender_initializes_receivers: ClassVar[bool] = False
    """The sender drives the inference-side init itself, so the worker must NOT
    also call ``init_weight_update_communicator``.

    False for the push backends: worker rank 0 pushes ``init_info`` to the
    servers, concurrently with ``create_sender`` (broadcast needs both sides in
    the same process group at once). True for a backend whose own engine owns the
    handshake."""

    sender_needs_weight_extractor: ClassVar[bool] = False
    """``create_sender`` takes the worker's ``weight_extractor``.

    Only for backends that must rendezvous at init rather than on the first send,
    which needs the extractor that early. The worker passes it only when this is
    True, so the other strategies' signatures are untouched."""

    groups_chunks_by_module: ClassVar[bool] = False
    """Whether the FSDP extractor should group parameters per module.

    Fused-weight loaders want one chunk per module (CUDA IPC); everything else
    wants one parameter per name. Read off the strategy class so the extractor's
    construction does not have to re-derive the backend."""

    @staticmethod
    @abstractmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig",
        inference_world_size: Optional[int] = None,
        base_model_path: Optional[str] = None,
    ) -> WeightSyncInitInfo:
        """Create init info with all config-derived args.

        Args:
            ie_cfg: Inference engine configuration.
            inference_world_size: Total number of inference workers (from
                ``client.get_world_size()``). Required by strategies that use it
                (broadcast); strategies that don't (CUDA IPC) ignore it.
            base_model_path: Policy model path.

        Returns:
            WeightSyncInitInfo containing all args needed for sender/receiver creation.
        """
        ...

    @staticmethod
    @abstractmethod
    def get_vllm_transfer_engine() -> type:
        """Return the vLLM weight-transfer engine class for this strategy.

        Broadcast -> ``NCCLWeightTransferEngine``; CUDA IPC ->
        ``IPCWeightTransferEngine``. This is the receive-side engine the
        inference servers drive natively. Currently unused on the sender side
        (we route through the SkyRL ``/collective_rpc`` wrap); kept as the
        canonical strategy->engine mapping.
        """
        ...

    @staticmethod
    @abstractmethod
    def create_sender(
        init_info: WeightSyncInitInfo,
        inference_client: "RemoteInferenceClient",
    ) -> WeightTransferSender:
        """Create a sender for the training worker side.

        This method must be called on all training ranks. Implementations may
        have different initialization logic for different ranks (e.g., only rank 0
        joins a process group for broadcast, while all ranks participate for IPC).

        Args:
            init_info: WeightSyncInitInfo containing config-derived args.
            inference_client: Client for coordinating with inference engines.

        Returns:
            A configured WeightTransferSender instance.
        """
        ...
