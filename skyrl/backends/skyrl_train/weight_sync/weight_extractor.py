"""Weight extractor interface for extracting weights from training backends."""

from abc import ABC, abstractmethod
from typing import Iterator

import torch

from .base import WeightChunk


class WeightExtractor(ABC):
    """Extracts weights from training backend models.

    Subclasses implement backend-specific logic to extract model weights,
    handle sharding, and prepare them for transfer to inference engines.
    """

    @abstractmethod
    def extract_weights(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        """Extract weights from the model as WeightChunk objects.

        Implementations should:
        - Gather sharded weights into full tensors
        - Convert tensors to the specified dtype for inference
        - Ensure tensors are contiguous in memory
        - Optionally group related parameters (e.g., QKV for efficiency)

        Args:
            dtype: Target dtype for inference (e.g., torch.bfloat16, torch.float16)

        Yields:
            WeightChunk objects containing model parameters ready for transfer
        """
        ...
