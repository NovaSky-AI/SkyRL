"""Weight transfer strategy abstractions for distributed RL training.

This module defines the abstract interfaces for transferring model weights
from training workers to inference engines. The strategy pattern allows different
transfer mechanisms (broadcast, CUDA IPC) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Optional

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk

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

    @abstractmethod
    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
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

    def bind_trainer_worker(self, worker) -> None:
        """Give the sender a handle to its training worker.

        Default no-op. The sharded-RDT sender overrides this: its send path drives
        gather/serve methods that live on the worker actor (the trainer produces
        the weights the inference workers pull), so it needs the worker handle.
        """
        return None


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

    @staticmethod
    @abstractmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig", inference_world_size: Optional[int] = None
    ) -> WeightSyncInitInfo:
        """Create init info with all config-derived args.

        Args:
            ie_cfg: Inference engine configuration.
            inference_world_size: Total number of inference workers (from
                ``client.get_world_size()``). Required by strategies that use it
                (broadcast); strategies that don't (CUDA IPC) ignore it.

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
    def populate_init_info(
        init_info: WeightSyncInitInfo, *, weight_extractor=None
    ) -> None:
        """Fill init-info fields that require the live model, before receiver init.

        Default no-op (broadcast / CUDA IPC send their metadata per-chunk at
        update time). The sharded-RDT strategy overrides this to fill
        names/dtypes/shapes from the weight extractor, since its receiver bakes a
        replay plan over the full name list at init.
        """
        return None

    @staticmethod
    def initialize_receivers(
        init_info: WeightSyncInitInfo,
        inference_client: "RemoteInferenceClient",
    ):
        """Initialize the inference-side receivers. Returns an awaitable.

        Default routes to vLLM's native weight-transfer init
        (``init_weight_update_communicator``), used by broadcast/CUDA IPC. The
        sharded-RDT strategy overrides this to drive its own init RPC. Called on
        rank 0 only, concurrently with ``create_sender`` (broadcast needs the two
        to join the same process group without deadlock).
        """
        return inference_client.init_weight_update_communicator(init_info)

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
