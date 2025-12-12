"""Weight transfer strategy abstractions for distributed RL training.

This module defines the abstract interfaces for transferring model weights
from training workers to inference engines. The strategy pattern allows different
transfer mechanisms (broadcast, CUDA IPC) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Tuple

import torch

from skyrl_train.inference_engines.base import NamedWeightsUpdateRequest
from skyrl_train.weight_sync.base import WeightChunk


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
    def receive_weights(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield (name, tensor) tuples by pulling data from transfer channel.

        Args:
            request: Weight update request with metadata (names, dtypes, shapes, extras).

        Yields:
            Tuples of (parameter_name, tensor) for each weight in the request.
        """
        ...


class WeightTransferStrategy(ABC):
    """Factory for creating transfer senders and receivers.

    Each strategy implementation encapsulates the configuration needed for a specific
    transfer mechanism. Strategy-specific args (rank, process group, etc.) are passed
    to the strategy constructor, not to create_sender/create_receiver.
    """

    @abstractmethod
    def create_sender(self) -> WeightTransferSender:
        """Create a sender for the training worker side."""
        ...

    @abstractmethod
    def create_receiver(self) -> WeightTransferReceiver:
        """Create a receiver for the inference engine side."""
        ...
