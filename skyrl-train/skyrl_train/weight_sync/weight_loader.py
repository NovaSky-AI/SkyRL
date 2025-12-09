"""Weight loader interface for inference engines."""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import torch


class WeightLoader(ABC):
    """Loads received weights into inference engine.

    Implementations are engine-specific (vLLM, SGLang, etc.) and handle
    the mechanics of applying weights to the inference model.
    """

    @abstractmethod
    async def load_weights(self, weights: Iterator[Tuple[str, torch.Tensor]]) -> None:
        """Load weights into the inference engine.

        Args:
            weights: Iterator yielding (param_name, tensor) tuples
                    Each tensor is already on the correct device and dtype.
        """
        ...
