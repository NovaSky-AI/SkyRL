"""Abstract backend interface for TinkerEngine.

Backends handle all model state and computation. The engine handles file I/O and database operations.

Design:
  1. AbstractBackend (backend.py)
     Clean interface defining what backends must implement:
     - create_optimizer
     - process_forward_backward_batch, process_forward_batch, process_optim_step, process_sample_batch
     - extract_checkpoint_data, insert_checkpoint_data (pure state manipulation)
     - extract_sampler_weights, insert_sampler_weights
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
     - Stores models and optimizers dicts
"""

from abc import ABC, abstractmethod

import jax
from flax import nnx

from tx.tinker import types
from tx.tinker.config import EngineConfig


class AbstractBackend(ABC):
    """Abstract base class for TinkerEngine backends.

    Backends handle computation and model state manipulation.
    Database operations are handled by TinkerEngine.
    """

    config: EngineConfig
    mesh: jax.sharding.Mesh
    model: nnx.Module
    metrics: types.EngineMetrics
    graphdef: nnx.GraphDef
    lora_params: nnx.State
    non_lora_params: nnx.State

    @abstractmethod
    def __init__(self, config: EngineConfig, **kwargs):
        """Initialize the backend."""
        pass

    @abstractmethod
    def create_optimizer(self, model_id: str) -> nnx.Optimizer:
        """Create an optimizer for a model.

        Args:
            model_id: The model identifier

        Returns:
            An nnx.Optimizer instance for the model
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
        request_data: types.OptimStepInput,
        optimizer: nnx.Optimizer,
        adapter_index: int,
    ) -> types.OptimStepOutput:
        """Process an optimizer step request.

        Args:
            model_id: The model identifier
            request_data: The optimizer step input parameters
            optimizer: The optimizer instance for this model
            adapter_index: The adapter index for this model

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
        optimizers: dict[str, nnx.Optimizer],
    ) -> None:
        """Save training checkpoint to disk.

        Args:
            output_path: Path to save the checkpoint
            model_id: The model identifier
            models: Dict mapping model_id to ModelMetadata
            optimizers: Dict mapping model_id to Optimizer
        """
        pass

    @abstractmethod
    def extract_checkpoint_data(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
        optimizers: dict[str, nnx.Optimizer],
    ) -> dict:
        """Extract model state for checkpointing.

        Args:
            model_id: The model identifier
            models: Dict mapping model_id to ModelMetadata
            optimizers: Dict mapping model_id to Optimizer

        Returns:
            Dictionary containing checkpoint data (weights, optimizer state, config).
        """
        pass

    @abstractmethod
    def insert_checkpoint_data(
        self,
        model_id: str,
        checkpoint_data: dict,
        models: dict[str, types.ModelMetadata],
        optimizers: dict[str, nnx.Optimizer],
    ) -> None:
        """Insert checkpoint data into model state.

        Args:
            model_id: The model identifier
            checkpoint_data: Dictionary from extract_checkpoint_data or loaded from disk
            models: Dict mapping model_id to ModelMetadata
            optimizers: Dict mapping model_id to Optimizer
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
    def extract_sampler_weights(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> dict:
        """Extract weights for sampler checkpoint.

        Args:
            model_id: The model identifier
            models: Dict mapping model_id to ModelMetadata

        Returns:
            Dictionary containing sampler weights data.
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
