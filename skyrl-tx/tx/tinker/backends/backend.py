"""Abstract backend interface for TinkerEngine.

Backends handle all model state, adapter management, and computation.
The engine handles database operations and scheduling.

Design:
  1. AbstractBackend (backend.py)
     Clean interface defining what backends must implement:
     - create_model (manages model metadata, adapter allocation, and optimizer lifecycle)
     - forward_backward, forward, optim_step, sample
     - load_checkpoint, save_checkpoint, save_sampler_checkpoint

  2. JaxBackend (jax.py)
     - Implements all abstract methods in Jax, fully supporting MultiLoRA for training and sampling
     - Uses jax.value_and_grad for gradient computation
     - Uses 2D mesh (fsdp, tp)
     - Multi-adapter AccumulatedGradients with counts array
     - Manages model metadata and adapter_index allocation internally

  3. TinkerEngine (engine.py)
     - Instantiates backend based on config
     - Delegates computation and model management to self.backend
     - Handles all database operations
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from tx.tinker import types


class AbstractBackend(ABC):
    """Abstract base class for TinkerEngine backends.

    Backends handle computation and model state manipulation.
    Database operations are handled by TinkerEngine.
    """

    @abstractmethod
    def __init__(self, base_model: str, config: BaseModel):
        """Initialize the backend."""
        pass

    @abstractmethod
    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        """Create a new model in the backend.

        Creates optimizer and configures LoRA adapter.

        Args:
            model_id: The model identifier
            lora_config: LoRA configuration with rank and alpha
        """
        pass

    @abstractmethod
    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Run forward and backward pass on a batch.

        Args:
            prepared_batch: PreparedModelPassBatch with all data extracted from requests

        Returns:
            Dict mapping request_id to result or error
        """
        pass

    @abstractmethod
    def forward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Run forward-only pass on a batch (no gradient computation).

        Args:
            prepared_batch: PreparedModelPassBatch with all data extracted from requests

        Returns:
            Dict mapping request_id to result or error
        """
        pass

    @abstractmethod
    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        """Apply an optimizer step using accumulated gradients.

        Args:
            model_id: The model identifier
            request_data: The optimizer step input parameters

        Returns:
            OptimStepOutput result
        """
        pass

    @abstractmethod
    def sample(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Generate samples for a batch of requests.

        Args:
            prepared_batch: PreparedSampleBatch with all data extracted from requests

        Returns:
            Dict mapping request_id to result or error
        """
        pass

    @abstractmethod
    def save_checkpoint(self, output_path, model_id: str) -> None:
        """Save training checkpoint to disk.

        Args:
            output_path: Path to save the checkpoint
            model_id: The model identifier
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        """Load training checkpoint from disk.

        Args:
            checkpoint_path: Path to the checkpoint file
            model_id: The model identifier
        """
        pass

    @abstractmethod
    def save_sampler_checkpoint(self, output_path, model_id: str, persist: bool = True) -> None:
        """Prepare model weights for sampling and optionally save to disk.

        Backends that use colocated inference engines should sync weights
        in-memory regardless of ``persist``.  When ``persist`` is *False*
        the backend may skip the expensive disk write and only place a
        lightweight marker at ``output_path``.

        Args:
            output_path: Path to save the checkpoint tar.gz file
            model_id: The model identifier
            persist: If True, write a full model snapshot to disk.
                     If False, only sync weights in-memory (hot path).
        """
        pass

    @abstractmethod
    def has_model(self, model_id: str) -> bool:
        """Check if a model is registered with the backend.

        Args:
            model_id: The model identifier

        Returns:
            True if the model is registered, False otherwise
        """
        pass

    @abstractmethod
    def delete_model(self, model_id: str) -> None:
        """Delete a model and free all associated resources.

        Args:
            model_id: The model identifier
        """
        pass
