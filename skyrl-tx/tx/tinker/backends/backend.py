"""Abstract backend interface for TinkerEngine.

Backends handle all model state and computation. The engine handles file I/O and database operations.

Design:
  1. AbstractBackend (backend.py)
     Clean interface defining what backends must implement:
     - register_model, unregister_model (optimizer lifecycle managed internally)
     - process_forward_backward_batch, process_forward_batch, process_optim_step, process_sample_batch
     - insert_checkpoint_data (pure state manipulation)
     - insert_sampler_weights
     - save_checkpoint, save_sampler_checkpoint (file I/O in backend)

  2. NativeBackend (native.py)
     - Implements all abstract methods fully for Qwen3 + LoRA
     - Uses jax.value_and_grad for gradient computation
     - Uses 2D mesh (dp, tp)
     - Multi-adapter AccumulatedGradients with counts array

  3. TinkerEngine (engine.py)
     - Instantiates backend based on config
     - Delegates computation to self.backend
     - Handles all database operations
     - Manages model metadata (backends manage optimizers internally)
"""

from abc import ABC, abstractmethod

from tx.tinker import types
from tx.tinker.config import EngineConfig


class AbstractBackend(ABC):
    """Abstract base class for TinkerEngine backends.

    Backends handle computation and model state manipulation.
    Database operations are handled by TinkerEngine.
    """

    @abstractmethod
    def __init__(self, config: EngineConfig, **kwargs):
        """Initialize the backend."""
        pass

    @abstractmethod
    def register_model(self, model_id: str, adapter_index: int, lora_config: types.LoraConfig) -> None:
        """Register a new model with the backend.

        Creates optimizer and configures LoRA adapter internally.

        Args:
            model_id: The model identifier
            adapter_index: The adapter slot index to use
            lora_config: LoRA configuration with rank and alpha
        """
        pass

    @abstractmethod
    def process_forward_backward_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward_backward requests in a batch.

        Args:
            prepared_batch: PreparedModelPassBatch with all data extracted from requests

        Returns:
            Dict mapping request_id to result or error
        """
        pass

    @abstractmethod
    def process_forward_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward-only requests in a batch (no gradient computation).

        Args:
            prepared_batch: PreparedModelPassBatch with all data extracted from requests

        Returns:
            Dict mapping request_id to result or error
        """
        pass

    @abstractmethod
    def process_optim_step(
        self,
        model_id: str,
        adapter_index: int,
        request_data: types.OptimStepInput,
    ) -> types.OptimStepOutput:
        """Process an optimizer step request.

        Args:
            model_id: The model identifier
            adapter_index: The adapter index for this model
            request_data: The optimizer step input parameters

        Returns:
            OptimStepOutput result
        """
        pass

    @abstractmethod
    def process_sample_batch(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Process multiple sample requests in a single batch.

        Args:
            prepared_batch: PreparedSampleBatch with all data extracted from requests

        Returns:
            Dict mapping request_id to result or error
        """
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        output_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Save training checkpoint to disk.

        Args:
            output_path: Path to save the checkpoint
            model_id: The model identifier
            models: Dict mapping model_id to ModelMetadata
        """
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Load training checkpoint from disk.

        Args:
            checkpoint_path: Path to the checkpoint file
            model_id: The model identifier
            models: Dict mapping model_id to ModelMetadata
        """
        pass

    @abstractmethod
    def insert_checkpoint_data(
        self,
        model_id: str,
        checkpoint_data: dict,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Insert checkpoint data into model state.

        Args:
            model_id: The model identifier
            checkpoint_data: Dictionary containing checkpoint data loaded from disk
            models: Dict mapping model_id to ModelMetadata
        """
        pass

    @abstractmethod
    def save_sampler_checkpoint(
        self,
        output_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Save sampler checkpoint to disk as tar.gz.

        Args:
            output_path: Path to save the checkpoint tar.gz file
            model_id: The model identifier
            models: Dict mapping model_id to ModelMetadata
        """
        pass

    @abstractmethod
    def insert_sampler_weights(
        self,
        model_id: str,
        checkpoint_id: str,
        checkpoint_path,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Insert sampler weights into model state from checkpoint file.

        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            checkpoint_path: Path to the checkpoint file
            models: Dict mapping model_id to ModelMetadata
        """
        pass
