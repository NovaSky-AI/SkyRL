"""Weight transfer strategy abstractions for distributed RL training.

This module defines the abstract interfaces for transferring model weights
from training workers to inference engines. The strategy pattern allows different
transfer mechanisms (broadcast, CUDA IPC) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Union

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.config import SkyRLTrainConfig


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


class WeightTransferStrategy(ABC):
    """Stateless factory for creating init info and senders.

    Each strategy implementation provides static methods to create:
    - init_info: Contains all config-derived args
    - sender: Uses init_info + inference_client

    Usage on sender side:
        init_info = Strategy.create_init_info(cfg)
        sender = Strategy.create_sender(init_info, inference_client)

    The receiver side lives inside the inference servers (vLLM's native weight
    transfer engine), driven via the inference client's HTTP control plane.
    """

    @staticmethod
    @abstractmethod
    def create_init_info(
        cfg: "Union[SkyRLTrainConfig, DictConfig]", inference_world_size: Optional[int] = None
    ) -> WeightSyncInitInfo:
        """Create init info with all config-derived args.

        Args:
            cfg: Configuration object containing generator settings.
            inference_world_size: Total number of inference workers (from client.get_world_size()).
                Used by HTTP inference path. Strategies that don't need world_size can ignore this.

        Returns:
            WeightSyncInitInfo containing all args needed for sender/receiver creation.
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
