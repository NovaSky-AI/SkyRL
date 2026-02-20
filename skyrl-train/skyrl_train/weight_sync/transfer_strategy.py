"""Weight transfer strategy abstractions for distributed RL training.

This module defines the abstract interfaces for transferring model weights
from training workers to inference engines. The strategy pattern allows different
transfer mechanisms (broadcast, CUDA IPC) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Tuple, Union

import torch

from skyrl_train.weight_sync.base import WeightChunk

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from skyrl_train.config import SkyRLConfig
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
    from skyrl_train.weight_sync.base import WeightUpdateRequest


@dataclass
class WeightSyncInitInfo(ABC):
    """Base class for weight sync initialization info."""

    override_existing_receiver: bool
    """Whether to override an existing weight receiver. If False and a receiver exists, init is skipped."""

    @staticmethod
    @abstractmethod
    def strategy_type() -> type:
        """Return the strategy class for this init info type."""
        ...

    def for_engine(self, engine_index: int, tp_size: int, pp_size: int) -> "WeightSyncInitInfo":
        """Return init_info adjusted for a specific engine.

        Override in subclasses that need per-engine adjustments (e.g., rank offset).
        Default implementation returns self unchanged.

        Args:
            engine_index: Index of the engine (0-based).
            tp_size: Tensor parallel size of the engine.
            pp_size: Pipeline parallel size of the engine.

        Returns:
            Adjusted init_info for the specific engine.
        """
        return self


class WeightTransferSender(ABC):
    """Strategy-specific component that sends WeightChunk data to inference actors.

    Implementations handle the transfer primitive (broadcast, CUDA IPC) and coordinate
    with inference actors.
    """

    @abstractmethod
    async def send_chunks(self, chunks: Iterable[WeightChunk]) -> None:
        """Send chunks using this transfer strategy.

        This method must be called on all training ranks. Implementations may have
        different behavior for different ranks.

        Args:
            chunks: Iterable of WeightChunk objects to send.
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources used by the sender (e.g., destroy process groups)."""
        ...


class WeightTransferReceiver(ABC):
    """Strategy-specific component that receives weights from training workers.

    Implementations handle receiving data via the transfer primitive (broadcast, CUDA IPC)
    and yielding tensors for the loader to apply to the model.
    """

    @abstractmethod
    def receive_weights(self, request: "WeightUpdateRequest") -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield (name, tensor) tuples by pulling data from transfer channel.

        This method must be called on all inference engine ranks. Implementations may have
        different behavior for different ranks.

        Args:
            request: Weight update request.

        Yields:
            Tuples of (parameter_name, tensor) for each weight in the request.
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources used by the receiver (e.g., destroy process groups)."""
        ...


class WeightTransferStrategy(ABC):
    """Stateless factory for creating init info, senders and receivers.

    Each strategy implementation provides static methods to create:
    - init_info: Contains all config-derived args
    - sender: Uses init_info + inference_client
    - receiver: Uses init_info + strategy-specific args

    Usage on sender side:
        init_info = Strategy.create_init_info(cfg)
        sender = Strategy.create_sender(init_info, inference_client)

    Usage on receiver side (for each engine):
        engine_init_info = init_info.for_engine(engine_index, tp_size, pp_size)
        receiver = engine_init_info.strategy_type().create_receiver(engine_init_info)
    """

    @staticmethod
    @abstractmethod
    def create_init_info(
        cfg: "Union[SkyRLConfig, DictConfig]", inference_world_size: Optional[int] = None
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
        inference_client: "InferenceEngineClient",
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

    @staticmethod
    @abstractmethod
    def create_receiver(init_info: WeightSyncInitInfo) -> WeightTransferReceiver:
        """Create a receiver for the inference engine side.

        This method must be called on all inference engine ranks. Implementations may
        have different initialization logic for different ranks.

        Args:
            init_info: WeightSyncInitInfo from the sender.

        Returns:
            A configured WeightTransferReceiver instance.
        """
        ...
