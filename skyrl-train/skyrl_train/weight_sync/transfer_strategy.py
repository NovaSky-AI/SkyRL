"""Weight transfer strategy abstractions for distributed RL training.

This module defines the abstract interfaces for transferring model weights
from training workers to inference engines. The strategy pattern allows different
transfer mechanisms (broadcast, CUDA IPC) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator, Tuple

import torch

from skyrl_train.weight_sync.base import WeightChunk

if TYPE_CHECKING:
    from skyrl_train.weight_sync.base import WeightUpdateRequest


@dataclass
class WeightSyncInitInfo(ABC):
    """Base class for weight sync initialization info."""

    @staticmethod
    @abstractmethod
    def strategy_type() -> type:
        """Return the strategy class for this init info type."""
        ...


class WeightTransferSender(ABC):
    """Strategy-specific component that sends WeightChunk data to inference actors.

    Implementations handle the transfer primitive (broadcast, CUDA IPC) and coordinate
    with inference actors.
    """

    @abstractmethod
    async def send_chunks(self, chunks: Iterable[WeightChunk]) -> None:
        """Send chunks using this transfer strategy.

        Args:
            chunks: Iterable of WeightChunk objects to send.
        """
        ...


class WeightTransferReceiver(ABC):
    """Strategy-specific component that receives weights from training workers.

    Implementations handle receiving data via the transfer primitive (broadcast, CUDA IPC)
    and yielding tensors for the loader to apply to the model.
    """

    @abstractmethod
    def receive_weights(self, request: "WeightUpdateRequest") -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield (name, tensor) tuples by pulling data from transfer channel.

        Args:
            request: Weight update request.

        Yields:
            Tuples of (parameter_name, tensor) for each weight in the request.
        """
        ...


class WeightTransferStrategy(ABC):
    """Stateless factory for creating init info, senders and receivers.

    Each strategy implementation provides static methods to create:
    - init_info: Contains all config-derived args
    - sender: Uses init_info + inference_client
    - receiver: Uses init_info + strategy-specific args

    Usage on sender side:
        init_info = Strategy.create_init_info(...)
        sender = Strategy.create_sender(init_info, inference_client)

    Usage on receiver side:
        receiver = init_info.strategy_type().create_receiver(init_info, ...)
    """

    @staticmethod
    @abstractmethod
    def create_init_info(cfg: "DictConfig") -> WeightSyncInitInfo:
        """Create init info with all config-derived args.

        Args:
            cfg: Configuration object containing generator settings.

        Returns:
            WeightSyncInitInfo containing all args needed for sender/receiver creation.
        """
        ...

    @staticmethod
    @abstractmethod
    def create_sender(
        init_info: WeightSyncInitInfo,
        inference_client: "InferenceEngineClient",
    ) -> WeightTransferSender:
        """Create a sender for the training worker side.

        Args:
            init_info: WeightSyncInitInfo containing config-derived args.
            inference_client: Client for coordinating with inference engines.

        Returns:
            A configured WeightTransferSender instance.
        """
        ...

    @staticmethod
    @abstractmethod
    def create_receiver(init_info: WeightSyncInitInfo) -> WeightTransferReceiver:
        """Create a receiver for the inference engine side.

        Args:
            init_info: WeightSyncInitInfo from the sender.

        Returns:
            A configured WeightTransferReceiver instance.
        """
        ...


