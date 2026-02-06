"""JAX LoRA backend for TinkerEngine.

This backend implements the full training and inference pipeline for models
with LoRA adapters. It uses jax.value_and_grad for gradient computation and supports
multiple LoRA adapters via the AccumulatedGradients dataclass.

In multi-host mode, process 0 (coordinator) runs the engine with JaxBackend,
which broadcasts commands to workers. Workers run separately using `run_worker()`
or by running this module directly with `python -m tx.tinker.backends.jax`.

Usage:
    # Coordinator (process 0) - runs the full engine:
    uv run -m tx.tinker.engine --base-model Qwen/Qwen3-8B --backend-config '{
        "coordinator_address": "localhost:7777",
        "num_processes": 2,
        ...
    }'

    # Workers (process 1+) - run only the worker loop (receives config from coordinator):
    uv run -m tx.tinker.backends.jax --coordinator-address localhost:7777 --num-processes 2 --process-id 1
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from cloudpathlib import AnyPath
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import optax
from flax import nnx
from flax.training import checkpoints
from pydantic import BaseModel, Field, TypeAdapter
from transformers import AutoTokenizer, PretrainedConfig

from tx.models.configs import Qwen3Config
from tx.layers.lora import clear_lora_adapter, init_lora_adapter
from tx.tinker import types
from tx.tinker.backends.backend import AbstractBackend
from tx.tinker.backends.utils import pad, pad_batch, pad_to_fsdp
from tx.tinker.loss_fns import LOSS_FUNCTIONS
from tx.utils.models import (
    get_dtype,
    get_model_class,
    load_safetensors,
    load_lora_checkpoint,
    save_lora_checkpoint,
    extract_adapter_state,
    insert_adapter_state,
    round_up_seq_len,
    resolve_model_path,
)
from tx.utils.log import logger


class JaxBackendConfig(BaseModel, extra="forbid"):
    """Configuration specific to the JAX backend."""

    max_lora_adapters: int = Field(default=32, description="Maximum number of LoRA adapters")
    max_lora_rank: int = Field(default=32, description="Maximum LoRA rank")
    tensor_parallel_size: int = Field(default=1, description="Tensor parallelism degree to use for the model")
    expert_parallel_size: int = Field(default=1, description="Expert parallelism degree for MoE layers")
    fully_sharded_data_parallel_size: int = Field(
        default=1, description="Fully sharded data parallelism degree for the model"
    )
    train_micro_batch_size: int = Field(
        default=0,
        description="Micro-batch size (measured in number of sequences) for gradient accumulation; 0 means disabled (use full batch)",
    )
    sample_max_num_sequences: int = Field(
        default=0,
        description="Maximum batch size (measured in number of sequences) for sampling/generation; 0 means disabled (use full batch)",
    )
    enforce_eager: bool = Field(default=False, description="Disable JAX JIT compilation")
    shard_attention_heads: bool = Field(
        default=True,
        description="Whether to shard attention linear layers (qkvo projections) across tensor parallel devices",
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Whether to use gradient checkpointing (full recomputation strategy)",
    )
    loss_chunk_size: int = Field(
        default=1024,
        description="Chunk size for cross-entropy loss computation. Reduces memory by avoiding full [B*T, V] logits materialization. Set to 0 to disable chunking.",
    )
    # Multi-node configuration
    coordinator_address: str | None = Field(
        default=None,
        description="JAX coordinator address (host:port) for multi-node training. If not set, runs in single-node mode.",
    )
    num_processes: int | None = Field(
        default=None,
        description="Total number of processes in the multi-node cluster",
    )


@jax.tree_util.register_dataclass
@dataclass
class AccumulatedGradients:
    """Stores accumulated gradients for all LoRA adapters."""

    grad_sum: nnx.State
    counts: jax.Array

    @classmethod
    def create(cls, lora_params: nnx.State, max_adapters: int) -> "AccumulatedGradients":
        """Initialize with zeros."""
        return cls(
            grad_sum=jax.tree.map(jnp.zeros_like, lora_params),
            counts=jnp.zeros((max_adapters,), dtype=jnp.int32),
        )

    def add(self, lora_grads: nnx.State, adapter_indices: jax.Array) -> "AccumulatedGradients":
        """Accumulate gradients and increment counts."""
        # Count occurrences of each adapter index in the batch
        batch_counts = jnp.bincount(adapter_indices, length=self.counts.shape[0])
        return AccumulatedGradients(
            grad_sum=jax.tree.map(lambda a, b: a + b, self.grad_sum, lora_grads),
            counts=self.counts + batch_counts,
        )

    def get_mean(self, adapter_index: jax.Array) -> nnx.State:
        """Compute mean gradients for a specific adapter, with zeros for all other adapters."""
        count = self.counts[adapter_index]
        return jax.tree.map(
            lambda g: jnp.zeros_like(g).at[adapter_index].set(g[adapter_index] / count.astype(g.dtype)),
            self.grad_sum,
        )

    def reset_adapter(self, adapter_index: jax.Array) -> "AccumulatedGradients":
        """Reset gradients and count for a specific adapter."""
        return AccumulatedGradients(
            grad_sum=jax.tree.map(lambda g: g.at[adapter_index].set(0.0), self.grad_sum),
            counts=self.counts.at[adapter_index].set(0),
        )


class JaxBackendImpl(AbstractBackend):
    """JAX backend implementation for models with LoRA adapters.

    This is the core implementation class. Use JaxBackend (the distributed wrapper)
    for multi-host coordination.

    This backend:
    - Uses jax.value_and_grad for gradient computation
    - Uses 2D mesh (fsdp, tp) for fully sharded data parallelism and tensor parallelism
    - Supports multiple LoRA adapters via AccumulatedGradients with counts array
    - Supports both FORWARD and FORWARD_BACKWARD request types
    """

    def __init__(self, base_model: str, config: JaxBackendConfig):
        """Initialize JAX LoRA backend."""
        self.base_model = base_model
        self.config = config
        self.metrics = types.EngineMetrics()

        # Initialize the shared base model with LoRA config
        checkpoint_path = resolve_model_path(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_config = PretrainedConfig.from_pretrained(checkpoint_path)
        self.model_config = Qwen3Config(
            base_config,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            shard_attention_heads=config.shard_attention_heads,
            loss_chunk_size=config.loss_chunk_size,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        model_class = get_model_class(self.model_config)

        # Create model and load weights
        self.mesh = jax.make_mesh(
            (
                config.fully_sharded_data_parallel_size,
                config.expert_parallel_size,
                config.tensor_parallel_size,
            ),
            ("fsdp", "ep", "tp"),
            axis_types=(jax.sharding.AxisType.Auto,) * 3,
        )
        with jax.set_mesh(self.mesh), nnx.use_eager_sharding(True):
            self.model = model_class(self.model_config, dtype=get_dtype(self.model_config.dtype), rngs=nnx.Rngs(0))
            load_safetensors(checkpoint_path, self.model_config, self.model)

            # Split model into LoRA and non-LoRA parameters
            self.graphdef, self.lora_params, self.non_lora_params = nnx.split(self.model, self.model.is_lora_param, ...)

            # Initialize adapter 0 with minimal config (required for base model sampling path)
            init_lora_adapter(self.model, adapter_index=0, lora_config=types.LoraConfig(rank=1, alpha=1.0, seed=0))

            # Initialize global accumulated gradients
            self.accumulated_grads = AccumulatedGradients.create(self.lora_params, config.max_lora_adapters)

        # Per-model optimizer storage (managed internally)
        self.optimizers: dict[str, nnx.Optimizer] = {}

        # Store LoRA model metadata (model_id -> metadata)
        self.models: dict[str, types.ModelMetadata] = {}

        logger.info(
            f"Initialized base model {base_model} with "
            f"max_lora_adapters={config.max_lora_adapters}, max_lora_rank={config.max_lora_rank}"
        )

        self._create_loss_and_grad_fn()

    def _micro_batch_size(self, total: int) -> int:
        """Return effective micro-batch size; 0/absent => disabled (use full fused batch)."""
        mb = self.config.train_micro_batch_size
        return total if mb <= 0 else max(1, min(mb, total))

    @contextmanager
    def _jit_timing_context(self, seq_len: int, mode: str):
        """Context manager to track JIT compilation times for different sequence lengths.

        Args:
            seq_len: The sequence length being compiled
            mode: Either 'train' or 'sample' to track separately
        """
        jit_times = self.metrics.train_seq_len_jit_times if mode == "train" else self.metrics.sample_seq_len_jit_times
        if not self.config.enforce_eager and seq_len not in jit_times:
            logger.info(f"JIT compiling for {mode} seq_len={seq_len} in progress...")
            start_time = time.time()
            yield
            elapsed = time.time() - start_time
            jit_times[seq_len] = elapsed
            logger.info(f"JIT compilation for {mode} seq_len={seq_len} took {elapsed:.2f}s")
        else:
            yield

    def _create_loss_and_grad_fn(self):
        """Compile and cache the loss function to avoid re-jitting on every call."""

        def _model_forward(
            graphdef: nnx.GraphDef,
            lora_params: nnx.State,
            non_lora_params: nnx.State,
            input_ids: jax.Array,
            attention_mask: jax.Array,
            adapter_indices: jax.Array,
            target_ids: jax.Array,
        ) -> jax.Array:
            """Forward pass and logprobs computation."""
            model = nnx.merge(graphdef, lora_params, non_lora_params)
            output = model(
                input_ids,
                attention_mask=attention_mask,
                adapter_indices=adapter_indices,
            )
            return model.compute_logprobs(output.last_hidden_state, target_ids, adapter_indices)

        if self.config.gradient_checkpointing:
            # Wrap the model forward call to use jax.checkpoint for gradient checkpointing
            # policy=None corresponds to full activation recomputation
            _model_forward = jax.checkpoint(_model_forward, policy=None)

        def loss_for_lora(
            lora_params: nnx.State,
            non_lora_params: nnx.State,
            input_ids: jax.Array,
            attention_mask: jax.Array,
            adapter_indices: jax.Array,
            target_ids: jax.Array,
            loss_mask: jax.Array,
            loss_fn_types: jax.Array,
            sampling_logprobs: jax.Array,
            advantages: jax.Array,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            target_logprobs = _model_forward(
                self.graphdef,
                lora_params,
                non_lora_params,
                input_ids,
                attention_mask,
                adapter_indices,
                target_ids,
            )

            def compute_loss_per_example(loss_fn_type, target_logprobs, loss_mask, sampling_logprobs, advantages):
                return jax.lax.switch(
                    loss_fn_type,
                    LOSS_FUNCTIONS,
                    target_logprobs,
                    loss_mask,
                    sampling_logprobs,
                    advantages,
                )

            per_token_losses = jax.vmap(compute_loss_per_example)(
                loss_fn_types,
                target_logprobs,
                loss_mask,
                sampling_logprobs,
                advantages,
            )

            per_seq_loss = per_token_losses.sum(axis=-1) / jnp.maximum(loss_mask.sum(axis=-1), 1e-9)
            # Return sum of losses (we'll divide gradients by per-adapter batch size later)
            return per_seq_loss.sum(), (target_logprobs, per_token_losses)

        # Only differentiate with respect to lora_params (argnums=0)
        loss_and_grad_fn = jax.value_and_grad(loss_for_lora, argnums=0, has_aux=True)

        def forward_only(
            accumulated_grads: AccumulatedGradients,
            lora_params: nnx.State,
            non_lora_params: nnx.State,
            input_ids: jax.Array,
            attention_mask: jax.Array,
            adapter_indices: jax.Array,
            target_ids: jax.Array,
            loss_mask: jax.Array,
            loss_fn_types: jax.Array,
            sampling_logprobs: jax.Array,
            advantages: jax.Array,
        ) -> tuple[AccumulatedGradients, jax.Array, jax.Array]:
            _, (target_logprobs, per_token_losses) = loss_for_lora(
                lora_params,
                non_lora_params,
                input_ids,
                attention_mask,
                adapter_indices,
                target_ids,
                loss_mask,
                loss_fn_types,
                sampling_logprobs,
                advantages,
            )
            return accumulated_grads, per_token_losses, target_logprobs

        def forward_backward_and_accumulate(
            accumulated_grads: AccumulatedGradients,
            lora_params: nnx.State,
            non_lora_params: nnx.State,
            input_ids: jax.Array,
            attention_mask: jax.Array,
            adapter_indices: jax.Array,
            target_ids: jax.Array,
            loss_mask: jax.Array,
            loss_fn_types: jax.Array,
            sampling_logprobs: jax.Array,
            advantages: jax.Array,
        ) -> tuple[AccumulatedGradients, jax.Array, jax.Array]:
            """Fused forward-backward-accumulate operation."""
            # Forward-backward
            (_, (target_logprobs, per_token_losses)), lora_grads = loss_and_grad_fn(
                lora_params,
                non_lora_params,
                input_ids,
                attention_mask,
                adapter_indices,
                target_ids,
                loss_mask,
                loss_fn_types,
                sampling_logprobs,
                advantages,
            )
            # Accumulate gradients
            new_accumulated_grads = accumulated_grads.add(lora_grads, adapter_indices)
            return new_accumulated_grads, per_token_losses, target_logprobs

        if self.config.enforce_eager:
            # Disable JIT compilation for debugging
            self._forward_backward_and_accumulate = forward_backward_and_accumulate
            self._forward = forward_only

        else:
            # Retrieve the sharding of lora and non_lora params and compute the sharding of inputs and outputs
            lora_shardings = jax.tree.map(
                lambda spec: jax.NamedSharding(self.mesh, spec), nnx.get_partition_spec(self.lora_params)
            )
            non_lora_shardings = jax.tree.map(
                lambda spec: jax.NamedSharding(self.mesh, spec), nnx.get_partition_spec(self.non_lora_params)
            )
            # Get sharding for AccumulatedGradients
            accumulated_grads_shardings = jax.tree.map(
                lambda spec: jax.NamedSharding(self.mesh, spec), nnx.get_partition_spec(self.accumulated_grads)
            )

            # Shard batch inputs along the FSDP axis (batch, seq_len)
            batch_sharded_2d = jax.NamedSharding(self.mesh, jax.P("fsdp", None))

            # JIT the fused function
            # Input order: input_ids, attention_mask, adapter_indices, target_ids,
            #              loss_mask, loss_fn_types, sampling_logprobs, advantages
            # All batch arrays are sharded along batch dimension
            batch_sharded_1d = jax.NamedSharding(self.mesh, jax.P("fsdp"))
            input_shardings = (
                batch_sharded_2d,  # input_ids
                batch_sharded_2d,  # attention_mask
                batch_sharded_1d,  # adapter_indices (sharded, bincount runs per-device)
                batch_sharded_2d,  # target_ids
                batch_sharded_2d,  # loss_mask
                batch_sharded_1d,  # loss_fn_types (sharded, used in vmap over batch)
                batch_sharded_2d,  # sampling_logprobs
                batch_sharded_2d,  # advantages
            )
            self._forward_backward_and_accumulate = jax.jit(
                forward_backward_and_accumulate,
                in_shardings=(accumulated_grads_shardings, lora_shardings, non_lora_shardings) + input_shardings,
                out_shardings=(accumulated_grads_shardings, batch_sharded_2d, batch_sharded_2d),
                donate_argnames=("accumulated_grads",),
            )
            self._forward = jax.jit(
                forward_only,
                in_shardings=(accumulated_grads_shardings, lora_shardings, non_lora_shardings) + input_shardings,
                out_shardings=(accumulated_grads_shardings, batch_sharded_2d, batch_sharded_2d),
            )

        # JIT-compiled function to compute full gradients and apply optimizer update
        def compute_grads_and_update(
            accumulated_grads: AccumulatedGradients,
            lora_params: nnx.State,
            optimizer: nnx.Optimizer,
            adapter_index: jax.Array,
        ) -> AccumulatedGradients:
            """Compute full gradients, apply optimizer update, and reset accumulated grads."""
            optimizer.update(lora_params, accumulated_grads.get_mean(adapter_index))
            return accumulated_grads.reset_adapter(adapter_index)

        if self.config.enforce_eager:
            self._compute_grads_and_update = compute_grads_and_update
        else:
            self._compute_grads_and_update = nnx.jit(compute_grads_and_update)

    def has_model(self, model_id: str) -> bool:
        """Check if a model is registered with the backend."""
        return model_id in self.models

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        """Create a new model in the backend.

        Creates optimizer and configures LoRA adapter. Allocates adapter_index internally.
        """
        # Allocate adapter index for this model_id (find first available slot)
        # Index 0 is reserved for base model, so user models use indices 1 to max_lora_adapters-1
        used_indices = {m.adapter_index for m in self.models.values()}
        available_indices = set(range(1, self.config.max_lora_adapters)) - used_indices
        if not available_indices:
            raise ValueError(f"Maximum number of LoRA adapters ({self.config.max_lora_adapters}) reached")
        adapter_index = min(available_indices)
        assert 1 <= adapter_index <= self.config.max_lora_adapters - 1

        # Validate rank doesn't exceed max
        if not (0 < lora_config.rank <= self.config.max_lora_rank):
            raise ValueError(f"LoRA rank {lora_config.rank} must be between 1 and {self.config.max_lora_rank}")

        # Store model metadata
        self.models[model_id] = types.ModelMetadata(
            adapter_index=adapter_index,
            lora_config=lora_config,
        )

        # Create optimizer
        with jax.set_mesh(self.mesh):
            tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
            self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.model.is_lora_param)

        # Configure adapter
        init_lora_adapter(self.model, adapter_index, lora_config)
        logger.info(f"Created model {model_id} with adapter_index={adapter_index}, config={lora_config}")

    def delete_model(self, model_id: str) -> None:
        """Delete a model and free all associated resources."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        # Get adapter index before deleting metadata
        adapter_index = self.models[model_id].adapter_index

        # Clear LoRA adapter weights
        with jax.set_mesh(self.mesh):
            clear_lora_adapter(self.model, adapter_index)

        # Delete optimizer
        del self.optimizers[model_id]

        # Delete model metadata
        del self.models[model_id]

        logger.info(f"Deleted model {model_id} (adapter_index={adapter_index})")

    def _model_pass(
        self,
        prepared_batch: types.PreparedModelPassBatch,
        model_pass_fn: Callable,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Common batch processing logic for forward-only and forward-backward operations.

        Args:
            prepared_batch: PreparedModelPassBatch with all data extracted from requests
            model_pass_fn: Callable to perform the model pass (forward or forward_backward)

        Returns:
            Dict mapping request_id to result_data or error info
        """
        if not prepared_batch.all_input_ids:
            return {}

        results = {}

        # Extract data from prepared batch
        all_input_ids = prepared_batch.all_input_ids
        all_targets = prepared_batch.all_targets
        all_token_weights = prepared_batch.all_token_weights
        all_sampling_logprobs = prepared_batch.all_sampling_logprobs
        all_advantages = prepared_batch.all_advantages
        all_loss_fn_types = prepared_batch.all_loss_fn_types
        request_batch_slices = prepared_batch.request_batch_slices

        # Convert model_ids to adapter_indices
        all_adapter_indices = [self.models[model_id].adapter_index for model_id in prepared_batch.all_model_ids]

        # Pad sequences to same length. Also bin it so the JIT has to compile fewer kernels.
        max_len = round_up_seq_len(max(len(seq) for seq in all_input_ids))

        input_ids = pad_batch(all_input_ids, max_len, np.int32)
        target_ids = pad_batch(all_targets, max_len, np.int32)
        adapter_indices = np.array(all_adapter_indices, dtype=np.int32)
        loss_fn_types = np.array(all_loss_fn_types, dtype=np.int32)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = pad_batch([[1] * len(seq) for seq in all_input_ids], max_len, np.int32)
        loss_mask = pad_batch(all_token_weights, max_len, np.float32)
        sampling_logprobs = pad_batch(all_sampling_logprobs, max_len, np.float32)
        advantages = pad_batch(all_advantages, max_len, np.float32)

        total_bs = int(input_ids.shape[0])
        micro_bs = self._micro_batch_size(total_bs)
        seq_lens = [len(seq) for seq in all_input_ids]

        # Collect full padded arrays on device, slice after transfer
        token_losses_device = []
        logprobs_device = []
        seq_len = input_ids.shape[1]

        # Sharding specs for batch inputs
        sharding_2d = jax.NamedSharding(self.mesh, jax.P("fsdp", None))
        sharding_1d = jax.NamedSharding(self.mesh, jax.P("fsdp"))
        fsdp_size = self.mesh.shape["fsdp"]

        with jax.set_mesh(self.mesh), self._jit_timing_context(seq_len, mode="train"):
            for mb_start in range(0, total_bs, micro_bs):
                mb_end = min(mb_start + micro_bs, total_bs)

                # Pad and shard the micro-batch inputs
                (
                    mb_input_ids,
                    mb_attention_mask,
                    mb_target_ids,
                    mb_loss_mask,
                    mb_sampling_logprobs,
                    mb_advantages,
                    mb_adapter_indices,
                    mb_loss_fn_types,
                ) = jax.device_put(
                    (
                        pad_to_fsdp(input_ids[mb_start:mb_end], fsdp_size),
                        pad_to_fsdp(attention_mask[mb_start:mb_end], fsdp_size),
                        pad_to_fsdp(target_ids[mb_start:mb_end], fsdp_size),
                        pad_to_fsdp(loss_mask[mb_start:mb_end], fsdp_size),
                        pad_to_fsdp(sampling_logprobs[mb_start:mb_end], fsdp_size),
                        pad_to_fsdp(advantages[mb_start:mb_end], fsdp_size),
                        pad_to_fsdp(adapter_indices[mb_start:mb_end], fsdp_size),
                        pad_to_fsdp(loss_fn_types[mb_start:mb_end], fsdp_size),
                    ),
                    (sharding_2d,) * 6 + (sharding_1d,) * 2,
                )

                self.accumulated_grads, per_token_losses, target_logprobs = model_pass_fn(
                    self.accumulated_grads,
                    self.lora_params,
                    self.non_lora_params,
                    mb_input_ids,
                    mb_attention_mask,
                    mb_adapter_indices,
                    mb_target_ids,
                    mb_loss_mask,
                    mb_loss_fn_types,
                    mb_sampling_logprobs,
                    mb_advantages,
                )
                # Slice back to original size (remove FSDP padding)
                token_losses_device.append(per_token_losses[: mb_end - mb_start])
                logprobs_device.append(target_logprobs[: mb_end - mb_start])

        # Gather results from all hosts before device_get
        if jax.process_count() > 1:
            token_losses_device = [multihost_utils.process_allgather(x, tiled=True) for x in token_losses_device]
            logprobs_device = [multihost_utils.process_allgather(x, tiled=True) for x in logprobs_device]

        # Single batched device-to-host transfer for all arrays
        token_losses_host, logprobs_host = jax.device_get((token_losses_device, logprobs_device))

        # Flatten microbatches and slice to actual sequence lengths
        token_losses_out = []
        logprobs_out = []
        idx = 0
        for mb_losses, mb_logprobs in zip(token_losses_host, logprobs_host):
            for i in range(mb_losses.shape[0]):
                token_losses_out.append(mb_losses[i, : seq_lens[idx]].astype(jnp.float32))
                logprobs_out.append(mb_logprobs[i, : seq_lens[idx]].astype(jnp.float32))
                idx += 1

        # Compute per-request results
        for request_id, _, start_idx, end_idx in request_batch_slices:
            loss_fn_outputs = []
            # Compute per-example losses
            for i in range(start_idx, end_idx):
                # Extract losses for this example's tokens
                token_losses = token_losses_out[i]
                token_logprobs = logprobs_out[i]
                loss_fn_outputs.append(
                    {
                        "elementwise_loss": {
                            "data": token_losses.tolist(),
                            "dtype": "float32",
                            "shape": [token_losses.shape[0]],
                        },
                        "logprobs": {
                            "data": token_logprobs.tolist(),
                            "dtype": "float32",
                            "shape": [token_logprobs.shape[0]],
                        },
                    }
                )

            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )

        return results

    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Run forward and backward pass on a batch."""
        return self._model_pass(prepared_batch, self._forward_backward_and_accumulate)

    def forward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Run forward-only pass on a batch (no gradient computation)."""
        return self._model_pass(prepared_batch, self._forward)

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        """Apply an optimizer step using accumulated gradients."""
        adapter_index = self.models[model_id].adapter_index
        optimizer = self.optimizers[model_id]

        # Check if we have any gradients accumulated (count > 0)
        if self.accumulated_grads.counts[adapter_index] == 0:
            logger.warning(f"No accumulated gradients for model {model_id}, skipping optimizer step")
            return types.OptimStepOutput()

        # Update hyperparameters from the request
        hp = optimizer.opt_state.hyperparams
        hp["learning_rate"][...] = request_data.adam_params.learning_rate
        hp["b1"][...] = request_data.adam_params.beta1
        hp["b2"][...] = request_data.adam_params.beta2
        hp["eps"][...] = request_data.adam_params.eps
        hp["weight_decay"][...] = request_data.adam_params.weight_decay

        # JIT-compiled: compute full gradients, apply optimizer update, and reset accumulated grads
        with jax.set_mesh(self.mesh):
            self.accumulated_grads = self._compute_grads_and_update(
                self.accumulated_grads,
                self.lora_params,
                optimizer,
                jnp.int32(adapter_index),
            )

        logger.info(f"Applied optimizer step for model {model_id} (adapter {adapter_index})")
        return types.OptimStepOutput()

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
        if not prepared_batch.all_prompts:
            return {}

        results = {}

        # Extract data from prepared batch
        all_prompts = prepared_batch.all_prompts
        all_sampling_params = prepared_batch.all_sampling_params
        request_batch_slices = prepared_batch.request_batch_slices
        needs_prompt_logprobs = prepared_batch.needs_prompt_logprobs

        # Load sampler weights and get adapter indices
        all_adapter_indices = self.load_sampler_weights(prepared_batch)

        total_batch_size = len(all_prompts)
        max_batch_size = (
            self.config.sample_max_num_sequences if self.config.sample_max_num_sequences > 0 else total_batch_size
        )
        # Collect generated sequences and prompt logprobs across batches
        all_sequences: list[types.GeneratedSequence] = []
        all_prompt_logprobs: list[list[float]] = []

        # Sharding specs for sampling inputs
        sharding_2d = jax.NamedSharding(self.mesh, jax.P("fsdp", None))
        sharding_1d = jax.NamedSharding(self.mesh, jax.P("fsdp"))

        with jax.set_mesh(self.mesh):
            model = nnx.merge(self.graphdef, self.lora_params, self.non_lora_params)
            for batch_start in range(0, total_batch_size, max_batch_size):
                batch_end = min(batch_start + max_batch_size, total_batch_size)
                batch_prompts = pad(all_prompts[batch_start:batch_end], max_batch_size, fill=[])
                batch_adapter_indices = pad(all_adapter_indices[batch_start:batch_end], max_batch_size, fill=0)
                sampling_params = pad(
                    all_sampling_params[batch_start:batch_end], max_batch_size, fill=all_sampling_params[batch_start]
                )

                # Pad sequences to same length within the batch to minimize memory usage.
                # Also bin it so the JIT has to compile fewer kernels.
                max_len = round_up_seq_len(max((len(seq) for seq in batch_prompts), default=0))
                input_ids = pad_batch(batch_prompts, max_len, np.int32)
                attention_mask = pad_batch([[1] * len(seq) for seq in batch_prompts], max_len, np.int32)

                # Shard inputs along FSDP axis (already padded to max_batch_size)
                input_ids, attention_mask, adapter_indices = jax.device_put(
                    (input_ids, attention_mask, np.array(batch_adapter_indices, dtype=np.int32)),
                    (sharding_2d, sharding_2d, sharding_1d),
                )

                with self._jit_timing_context(max_len, mode="sample"):
                    result = model.generate(
                        input_ids,
                        attention_mask,
                        sampling_params=sampling_params,
                        adapter_indices=adapter_indices,
                        prompt_logprobs=needs_prompt_logprobs,
                        tokenizer=self.tokenizer,
                    )
                # Only take the actual results, not the padded ones
                batch_size = batch_end - batch_start
                all_sequences.extend(
                    types.GeneratedSequence(stop_reason=stop_reason, tokens=tokens, logprobs=logprobs)
                    for stop_reason, tokens, logprobs in zip(
                        result.stop_reasons[:batch_size],
                        result.generated_ids[:batch_size],
                        result.logprobs[:batch_size],
                    )
                )
                if needs_prompt_logprobs and result.prompt_logprobs:
                    all_prompt_logprobs.extend(result.prompt_logprobs[:batch_size])

        for request_id, _, start_idx, end_idx, prompt_logprobs_requested in request_batch_slices:
            sequences = [all_sequences[i] for i in range(start_idx, end_idx)]
            # Each of `num_samples` samples in a request share the same prompt; use the first's prompt logprobs
            prompt_logprobs = (
                all_prompt_logprobs[start_idx] if prompt_logprobs_requested and all_prompt_logprobs else None
            )
            results[request_id] = types.SampleOutput(sequences=sequences, prompt_logprobs=prompt_logprobs)

        return results

    def save_checkpoint(self, output_path: AnyPath, model_id: str) -> None:
        """Save training checkpoint using Flax checkpoints."""
        checkpoint_data = self._extract_checkpoint_data(model_id)
        checkpoints.save_checkpoint_multiprocess(
            target=checkpoint_data,
            ckpt_dir=output_path,
            step=0,
            prefix="checkpoint_",
            overwrite=True,
        )
        logger.info(f"Saved training checkpoint to {output_path}")

    def _extract_checkpoint_data(self, model_id: str) -> dict:
        """Extract adapter state and optimizer state for checkpointing."""
        adapter_index = self.models[model_id].adapter_index
        rank = self.models[model_id].lora_config.rank
        lora_weights = extract_adapter_state(adapter_index, self.lora_params, rank)
        optimizer_state = extract_adapter_state(adapter_index, nnx.state(self.optimizers[model_id]), rank)
        return {
            "lora_weights": lora_weights,
            "optimizer_state": optimizer_state,
            "lora_config": self.models[model_id].lora_config.model_dump(),
        }

    def _insert_checkpoint_data(self, model_id: str, checkpoint_data: dict) -> None:
        """Insert checkpoint data into model state."""
        adapter_index = self.models[model_id].adapter_index
        rank = checkpoint_data["lora_config"]["rank"]

        if self.models[model_id].lora_config.rank != rank:
            raise ValueError(
                f"Rank mismatch: checkpoint has rank {rank}, "
                f"model configured with rank {self.models[model_id].lora_config.rank}"
            )

        insert_adapter_state(adapter_index, self.lora_params, checkpoint_data["lora_weights"], rank)
        insert_adapter_state(
            adapter_index, nnx.state(self.optimizers[model_id]), checkpoint_data["optimizer_state"], rank
        )

    def load_checkpoint(self, checkpoint_path: AnyPath, model_id: str) -> None:
        """Load training checkpoint using Flax checkpoints."""
        checkpoint = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path,
            target=self._extract_checkpoint_data(model_id),
            prefix="checkpoint_",
        )

        if checkpoint is None:
            raise FileNotFoundError(f"Training checkpoint not found in {checkpoint_path}")

        self._insert_checkpoint_data(model_id, checkpoint)
        logger.info(f"Loaded training checkpoint from {checkpoint_path}")

    def save_sampler_checkpoint(self, output_path: AnyPath, model_id: str, persist: bool = True) -> None:
        """Save sampler checkpoint as tar.gz using save_lora_checkpoint."""
        lora_model = self.models[model_id]
        save_lora_checkpoint(
            self.model,
            self.base_model,
            lora_model.lora_config,
            lora_model.adapter_index,
            output_path,
        )
        logger.info(f"Saved LoRA sampler checkpoint to {output_path}")

    def load_sampler_checkpoint(self, model_id: str, checkpoint_id: str, checkpoint_path: AnyPath) -> None:
        """Insert sampler weights from checkpoint file."""
        adapter_index = self.models[model_id].adapter_index
        adapter_config = self.models[model_id].lora_config
        load_lora_checkpoint(self.model, adapter_config, adapter_index, checkpoint_path)
        self.models[model_id].loaded_checkpoint_id = checkpoint_id
        logger.info(f"Loaded LoRA sampler weights for model {model_id} at adapter index {adapter_index}")

    def load_sampler_weights(self, prepared_batch: types.PreparedSampleBatch) -> list[int]:
        """Load sampler weights for all requests and return adapter indices array.

        Ensures all required checkpoints are loaded before sampling.

        Args:
            prepared_batch: PreparedSampleBatch with model_ids, checkpoint_ids, and other batch data

        Returns:
            The adapter_indices array for LoRA sampling [batch_size]
            Uses adapter index 0 for base model sampling (no LoRA)
        """
        adapter_indices = []
        loaded_adapters = set()  # Track adapters already used in this batch

        for model_id, checkpoint_id, checkpoint_path in zip(
            prepared_batch.all_model_ids, prepared_batch.all_checkpoint_ids, prepared_batch.all_checkpoint_paths
        ):
            if model_id:
                # This code path is for sampling from a LoRA adapter
                assert checkpoint_id != "", "checkpoint_id must be not empty"

                adapter_index = self.models[model_id].adapter_index
                if self.models[model_id].loaded_checkpoint_id == checkpoint_id:
                    # Weights already loaded in RAM
                    adapter_indices.append(adapter_index)
                else:
                    # Need to load from disk
                    assert adapter_index not in loaded_adapters, "Cannot override already used adapter"

                    logger.info(f"Loading LoRA sampler checkpoint from {checkpoint_path}")
                    self.load_sampler_checkpoint(model_id, checkpoint_id, AnyPath(checkpoint_path))
                    adapter_indices.append(adapter_index)

                loaded_adapters.add(adapter_index)
            else:
                # This code path is for sampling from the base model
                adapter_indices.append(0)

        return adapter_indices


# =============================================================================
# Multi-host coordination
# =============================================================================


class RpcPayload(BaseModel):
    """Generic RPC payload container using runtime type introspection.

    Instead of defining separate command classes for each method, this single
    generic container holds the method name and raw kwargs. The worker uses
    type hints from the target method to automatically re-hydrate the kwargs
    into the correct Pydantic models.
    """

    method: str
    kwargs: dict[str, Any]  # Contains raw dicts/JSON types


RpcPayloadAdapter: TypeAdapter[RpcPayload] = TypeAdapter(RpcPayload)


def _broadcast_command(cmd: RpcPayload | None) -> RpcPayload:
    """Broadcast an RpcPayload from coordinator to all workers using JSON.

    On coordinator (process 0): serializes and broadcasts the payload.
    On workers: receives and deserializes the payload (pass None).
    """
    if jax.process_index() == 0:
        assert cmd is not None, "Coordinator must provide a command to broadcast."
        data = RpcPayloadAdapter.dump_json(cmd)
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

    return RpcPayloadAdapter.validate_json(data_arr.tobytes())


class JaxBackend(JaxBackendImpl):
    """Distributed wrapper that broadcasts commands before calling JaxBackendImpl methods.

    Workers use runtime type introspection to re-hydrate arguments automatically.
    """

    def __init__(self, base_model: str, config: JaxBackendConfig):
        if config.coordinator_address is not None:
            jax.distributed.initialize(
                coordinator_address=config.coordinator_address,
                num_processes=config.num_processes,
                process_id=0,
            )
            logger.info(
                f"JAX distributed initialized: process_id={jax.process_index()} ({jax.process_count()} total), "
                f"local devices: {jax.local_device_count()}, total devices: {jax.device_count()}"
            )

        self._broadcast_and_call("__init__", base_model=base_model, config=config)

    def _broadcast_and_call(self, method: str, **kwargs):
        """Broadcast method call to workers and execute locally via super()."""
        if jax.process_count() > 1:
            hints = get_type_hints(getattr(JaxBackendImpl, method))

            # TODO: Remove AnyPath special case once https://github.com/drivendataorg/cloudpathlib/issues/537 is released
            def serialize(k, v):
                if hints.get(k) is AnyPath:
                    return str(v)
                return TypeAdapter(hints[k]).dump_python(v, mode="json") if k in hints else v

            _broadcast_command(RpcPayload(method=method, kwargs={k: serialize(k, v) for k, v in kwargs.items()}))
        return getattr(super(), method)(**kwargs)

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        self._broadcast_and_call("create_model", model_id=model_id, lora_config=lora_config)

    def forward_backward(self, prepared_batch: types.PreparedModelPassBatch):
        return self._broadcast_and_call("forward_backward", prepared_batch=prepared_batch)

    def forward(self, prepared_batch: types.PreparedModelPassBatch):
        return self._broadcast_and_call("forward", prepared_batch=prepared_batch)

    def optim_step(self, model_id: str, request_data: types.OptimStepInput):
        return self._broadcast_and_call("optim_step", model_id=model_id, request_data=request_data)

    def sample(self, prepared_batch: types.PreparedSampleBatch):
        return self._broadcast_and_call("sample", prepared_batch=prepared_batch)

    def save_checkpoint(self, output_path: AnyPath, model_id: str) -> None:
        self._broadcast_and_call("save_checkpoint", output_path=output_path, model_id=model_id)

    def load_checkpoint(self, checkpoint_path: AnyPath, model_id: str) -> None:
        self._broadcast_and_call("load_checkpoint", checkpoint_path=checkpoint_path, model_id=model_id)

    def save_sampler_checkpoint(self, output_path: AnyPath, model_id: str, persist: bool = True) -> None:
        # Write probe so workers can detect shared filesystem
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.with_name(output_path.name + ".probe").write_text("write_probe")
        self._broadcast_and_call("save_sampler_checkpoint", output_path=output_path, model_id=model_id, persist=persist)


def run_worker(coordinator_address: str, num_processes: int, process_id: int):
    """Entry point for worker processes.

    Initializes JAX distributed, receives config from coordinator, then runs
    the worker loop using runtime type introspection to re-hydrate arguments.

    Args:
        coordinator_address: JAX coordinator address (host:port)
        num_processes: Total number of processes in the cluster
        process_id: This process's ID (must be > 0 for workers)
    """
    if process_id == 0:
        raise ValueError("Worker process_id must be > 0 (process 0 is the coordinator)")

    # Initialize JAX distributed first (before any other JAX operations)
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )
    logger.info(
        f"Worker process_id={jax.process_index()} ({jax.process_count()} total) initialized, waiting for config from coordinator..."
    )

    # Receive INIT payload with base_model and config from coordinator
    init_payload = _broadcast_command(None)
    assert init_payload.method == "__init__", f"Expected __init__, got {init_payload.method}"
    config = JaxBackendConfig.model_validate(init_payload.kwargs["config"])
    logger.info(f"Worker received config: base_model={init_payload.kwargs['base_model']}, config={config}")

    backend = JaxBackendImpl(init_payload.kwargs["base_model"], config)

    logger.info(f"Worker process_id={jax.process_index()} entering command loop")

    while True:
        payload: RpcPayload = _broadcast_command(None)

        if not hasattr(backend, payload.method):
            logger.error(f"Unknown method: {payload.method}")
            continue

        method = getattr(backend, payload.method)

        # Re-hydrate raw dicts into Pydantic models using type hints
        hints = get_type_hints(method)
        kwargs = {k: TypeAdapter(hints[k]).validate_python(v) if k in hints else v for k, v in payload.kwargs.items()}
        method(**kwargs)


def main():
    """Entry point for running as a worker process."""
    import argparse

    parser = argparse.ArgumentParser(description="SkyRL tx tinker worker process")
    parser.add_argument(
        "--coordinator-address",
        required=True,
        help="JAX coordinator address (host:port)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        required=True,
        help="Total number of processes in the cluster",
    )
    parser.add_argument(
        "--process-id",
        type=int,
        required=True,
        help="This process's ID (must be > 0 for workers)",
    )

    args = parser.parse_args()
    run_worker(args.coordinator_address, args.num_processes, args.process_id)


if __name__ == "__main__":
    main()
