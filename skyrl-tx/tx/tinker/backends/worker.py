"""Multi-host distributed training backend wrapper.

In multi-host mode, process 0 (coordinator) runs the engine loop that polls the database
and dispatches work. All other processes run a worker loop that waits for commands
broadcast from the coordinator and call the same backend methods.

This pattern ensures all processes enter collective operations (broadcast/gather) in
the same order, avoiding deadlocks that would occur if each process independently
polled the database.

Usage:
    # Instead of using JaxBackend directly, use DistributedJaxBackend:
    backend = DistributedJaxBackend(base_model, config)

    # On coordinator (process 0): use normally, it broadcasts to workers
    # On workers: automatically enters worker loop and never returns
"""

import pickle
from enum import IntEnum

import numpy as np
import jax
from jax.experimental import multihost_utils

from tx.tinker import types
from tx.tinker.backends.backend import AbstractBackend
from tx.tinker.backends.jax import JaxBackend, JaxBackendConfig
from tx.utils.log import logger


class CommandType(IntEnum):
    """Command types broadcast from coordinator to workers."""

    NOOP = 0
    FORWARD_BACKWARD = 1
    FORWARD = 2
    SAMPLE = 3
    CREATE_MODEL = 4
    OPTIM_STEP = 5
    LOAD_CHECKPOINT = 6
    SAVE_CHECKPOINT = 7
    SAVE_SAMPLER_CHECKPOINT = 8
    SHUTDOWN = 99


def _broadcast_command_type(cmd: CommandType) -> CommandType:
    """Broadcast command type from coordinator to all workers."""
    arr = np.array([cmd], dtype=np.int32)
    arr = multihost_utils.broadcast_one_to_all(arr)
    return CommandType(arr[0])


def _broadcast_object(obj) -> object:
    """Broadcast a Python object from coordinator to all workers using pickle.

    On coordinator (process 0): serializes and broadcasts the object.
    On workers: receives and deserializes the object.
    """
    if jax.process_index() == 0:
        data = pickle.dumps(obj)
        size = np.array([len(data)], dtype=np.int64)
    else:
        size = np.array([0], dtype=np.int64)

    # Broadcast size first
    size = multihost_utils.broadcast_one_to_all(size)

    # Broadcast data
    if jax.process_index() == 0:
        data_arr = np.frombuffer(data, dtype=np.uint8)
    else:
        data_arr = np.zeros(size[0], dtype=np.uint8)

    data_arr = multihost_utils.broadcast_one_to_all(data_arr)

    return pickle.loads(data_arr.tobytes())


class DistributedJaxBackend(AbstractBackend):
    """Distributed wrapper around JaxBackend for multi-host coordination.

    This backend wraps JaxBackend to handle multi-host coordination:
    - On coordinator (process 0): broadcasts commands before calling JaxBackend methods
    - On workers (process != 0): enters worker loop on init and never returns

    The engine only runs on process 0 and uses this backend like a normal JaxBackend.
    Workers automatically participate in collective operations when commands are broadcast.
    """

    def __init__(self, base_model: str, config: JaxBackendConfig):
        """Initialize the multi-host backend.

        On coordinator: creates the underlying JaxBackend.
        On workers: creates JaxBackend then enters worker loop (blocking).
        """
        self._backend = JaxBackend(base_model, config)

        # Workers enter the command loop and never return
        if jax.process_count() > 1 and jax.process_index() != 0:
            self._worker_loop()
            # Worker loop only exits on shutdown - raise to prevent further execution
            raise SystemExit(0)

    @property
    def config(self) -> JaxBackendConfig:
        """Pass-through to underlying backend config."""
        return self._backend.config

    @property
    def metrics(self) -> types.EngineMetrics:
        """Pass-through to underlying backend metrics."""
        return self._backend.metrics

    def _broadcast_and_call(self, cmd: CommandType, method_name: str, *args):
        """Broadcast command and arguments, then call the underlying method."""
        if jax.process_count() > 1:
            _broadcast_command_type(cmd)
            _broadcast_object(args)
        return getattr(self._backend, method_name)(*args)

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        """Create a new model, broadcasting to workers in multi-host mode."""
        self._broadcast_and_call(CommandType.CREATE_MODEL, "create_model", model_id, lora_config)

    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Run forward and backward pass, broadcasting to workers in multi-host mode."""
        return self._broadcast_and_call(CommandType.FORWARD_BACKWARD, "forward_backward", prepared_batch)

    def forward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Run forward-only pass, broadcasting to workers in multi-host mode."""
        return self._broadcast_and_call(CommandType.FORWARD, "forward", prepared_batch)

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        """Apply optimizer step, broadcasting to workers in multi-host mode."""
        return self._broadcast_and_call(CommandType.OPTIM_STEP, "optim_step", model_id, request_data)

    def sample(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Generate samples, broadcasting to workers in multi-host mode."""
        return self._broadcast_and_call(CommandType.SAMPLE, "sample", prepared_batch)

    def save_checkpoint(self, output_path, model_id: str) -> None:
        """Save training checkpoint, broadcasting to workers in multi-host mode."""
        self._broadcast_and_call(CommandType.SAVE_CHECKPOINT, "save_checkpoint", output_path, model_id)

    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        """Load training checkpoint, broadcasting to workers in multi-host mode."""
        self._broadcast_and_call(CommandType.LOAD_CHECKPOINT, "load_checkpoint", checkpoint_path, model_id)

    def save_sampler_checkpoint(self, output_path, model_id: str) -> None:
        """Save sampler checkpoint, broadcasting to workers in multi-host mode."""
        self._broadcast_and_call(CommandType.SAVE_SAMPLER_CHECKPOINT, "save_sampler_checkpoint", output_path, model_id)

    def load_sampler_checkpoint(self, model_id: str, checkpoint_id: str, checkpoint_path) -> None:
        """Load sampler checkpoint - delegates to underlying backend."""
        # This is called internally by JaxBackend.sample(), not directly by engine
        self._backend.load_sampler_checkpoint(model_id, checkpoint_id, checkpoint_path)

    def has_model(self, model_id: str) -> bool:
        """Check if model is registered - local check only, no broadcast needed."""
        return self._backend.has_model(model_id)

    def _worker_loop(self):
        """Worker loop - wait for commands from coordinator and execute them.

        This method blocks until the coordinator sends a SHUTDOWN command.
        """
        logger.info(f"Worker {jax.process_index()} entering command loop")

        while True:
            # Wait for command type from coordinator
            cmd = _broadcast_command_type(CommandType.NOOP)

            if cmd == CommandType.SHUTDOWN:
                logger.info(f"Worker {jax.process_index()} received shutdown command")
                break
            elif cmd == CommandType.NOOP:
                continue
            elif cmd == CommandType.FORWARD_BACKWARD:
                (prepared_batch,) = _broadcast_object(None)
                self._backend.forward_backward(prepared_batch)
            elif cmd == CommandType.FORWARD:
                (prepared_batch,) = _broadcast_object(None)
                self._backend.forward(prepared_batch)
            elif cmd == CommandType.SAMPLE:
                (prepared_batch,) = _broadcast_object(None)
                self._backend.sample(prepared_batch)
            elif cmd == CommandType.CREATE_MODEL:
                model_id, lora_config = _broadcast_object(None)
                self._backend.create_model(model_id, lora_config)
            elif cmd == CommandType.OPTIM_STEP:
                model_id, request_data = _broadcast_object(None)
                self._backend.optim_step(model_id, request_data)
            elif cmd == CommandType.LOAD_CHECKPOINT:
                checkpoint_path, model_id = _broadcast_object(None)
                self._backend.load_checkpoint(checkpoint_path, model_id)
            elif cmd == CommandType.SAVE_CHECKPOINT:
                output_path, model_id = _broadcast_object(None)
                self._backend.save_checkpoint(output_path, model_id)
            elif cmd == CommandType.SAVE_SAMPLER_CHECKPOINT:
                output_path, model_id = _broadcast_object(None)
                self._backend.save_sampler_checkpoint(output_path, model_id)
            else:
                logger.warning(f"Worker received unknown command type: {cmd}")

        logger.info(f"Worker {jax.process_index()} exiting command loop")
