"""Background engine for processing training requests."""

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from pydantic import BaseModel
from sqlmodel import create_engine, Session, select, update, func

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import checkpoints


import optax
from transformers import PretrainedConfig

from tx.models.configs import Qwen3Config
from tx.tinker.db_models import FutureDB, RequestStatus, CheckpointDB, CheckpointStatus
from tx.tinker import types
from tx.tinker.config import EngineConfig, add_model
from tx.tinker.loss_fns import LOSS_TYPES, LOSS_FUNCTIONS
from tx.utils.storage import download_and_unpack, pack_and_upload
from tx.utils.models import (
    get_dtype,
    get_model_class,
    save_lora_checkpoint,
    load_lora_checkpoint,
    load_safetensors,
    extract_adapter_state,
    insert_adapter_state,
    round_up_seq_len,
    resolve_model_path,
)
from tx.layers.lora import update_adapter_config
from tx.utils.log import logger


@contextmanager
def log_timing(request: str):
    """Context manager to log execution time for a request."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.info(f"(timing) {request} took {elapsed:.3f}s")


def pad(xs, pad_to: int, *, fill):
    """Pad a list to a specified length with a fill value."""
    return xs + ([fill] * (pad_to - len(xs)))


def pad_batch(sequences: list[list], max_length: int, dtype, left: bool = False) -> jax.Array:
    """Pad a batch of sequences to max_length.

    Args:
        sequences: List of sequences to pad.
        max_length: Target length for all sequences.
        dtype: NumPy dtype for the output array.
        left: If True, use left-padding (tokens at end). Required for autoregressive
            generation so the last position corresponds to the last real token.
            If False (default), use right-padding (tokens at start).

    Returns:
        A JAX array of shape (batch_size, max_length) with the padded sequences.
    """
    batch_size = len(sequences)
    padded = np.zeros((batch_size, max_length), dtype=dtype)
    for i, seq in enumerate(sequences):
        assert len(seq) <= max_length, f"Sequence length {len(seq)} exceeds max_length {max_length}"
        if left:
            padded[i, max_length - len(seq) :] = seq
        else:
            padded[i, : len(seq)] = seq
    return jnp.asarray(padded)


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


class TinkerEngine:
    """Background engine for processing training requests."""

    def __init__(
        self,
        config: EngineConfig,
    ):
        """Initialize the engine with a database connection and base model."""
        self.config = config
        self.db_engine = create_engine(config.database_url, echo=False)

        # Store LoRA model metadata (model_id -> metadata)
        self.models: dict[str, types.ModelMetadata] = {}
        # Store optimizer instances per LoRA adapter (model_id -> optimizer)
        self.optimizers: dict[str, nnx.Optimizer] = {}
        # Metrics recorded in the engine
        self.metrics = types.EngineMetrics()

        # Initialize the shared base model with LoRA config
        checkpoint_path = resolve_model_path(self.config.base_model)
        base_config = PretrainedConfig.from_pretrained(checkpoint_path)
        self.model_config = Qwen3Config(
            base_config,
            max_lora_adapters=self.config.max_lora_adapters,
            max_lora_rank=self.config.max_lora_rank,
            shard_attention_heads=self.config.shard_attention_heads,
        )

        model_class = get_model_class(self.model_config)

        # Create model and load weights
        self.mesh = jax.make_mesh((1, self.config.tensor_parallel_size), ("dp", "tp"))
        with jax.set_mesh(self.mesh):
            self.model = model_class(self.model_config, dtype=get_dtype(self.model_config.dtype), rngs=nnx.Rngs(0))
            load_safetensors(checkpoint_path, self.model_config, self.model)

            # Split model into LoRA and non-LoRA parameters
            self.graphdef, self.lora_params, self.non_lora_params = nnx.split(self.model, self.model.is_lora_param, ...)
            update_adapter_config(self.model, adapter_index=0, lora_config=types.LoraConfig(rank=1, alpha=1.0))

            # Initialize global accumulated gradients
            self.accumulated_grads = AccumulatedGradients.create(self.lora_params, self.config.max_lora_adapters)

        logger.info(
            f"Initialized base model {self.config.base_model} with max_lora_adapters={self.config.max_lora_adapters}, max_lora_rank={self.config.max_lora_rank}"
        )

        self._create_loss_and_grad_fn()

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

    @contextmanager
    def _checkpoint_status_context(self, model_id: str, checkpoint_id: str, checkpoint_type: types.CheckpointType):
        """Context manager to handle checkpoint DB status updates.

        Fetches the checkpoint entry, yields it, and updates its status to COMPLETED
        or FAILED based on whether an exception occurred.
        """
        with Session(self.db_engine) as session:
            checkpoint_db = session.get(CheckpointDB, (model_id, checkpoint_id, checkpoint_type))
            if checkpoint_db is None:
                raise ValueError(
                    f"Checkpoint entry not found for model '{model_id}', checkpoint '{checkpoint_id}', type '{checkpoint_type}'"
                )

            try:
                yield checkpoint_db
                checkpoint_db.status = CheckpointStatus.COMPLETED
            except Exception as e:
                logger.exception(f"Error saving checkpoint for model {model_id}, checkpoint {checkpoint_id}: {e}")
                checkpoint_db.status = CheckpointStatus.FAILED
                checkpoint_db.error_message = str(e)
                raise
            finally:
                checkpoint_db.completed_at = datetime.now(timezone.utc)
                session.add(checkpoint_db)
                session.commit()

    def _create_loss_and_grad_fn(self):
        """Compile and cache the loss function to avoid re-jitting on every call."""

        # Wrap the model forward call to use nnx.remat for gradient checkpointing
        def _model_forward(
            graphdef: nnx.GraphDef,
            lora_params: nnx.State,
            non_lora_params: nnx.State,
            input_ids: jax.Array,
            attention_mask: jax.Array,
            adapter_indices: jax.Array,
        ) -> jax.Array:
            model = nnx.merge(graphdef, lora_params, non_lora_params)
            output = model(input_ids, attention_mask=attention_mask, adapter_indices=adapter_indices)
            return output.logits

        if self.config.gradient_checkpointing:
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
            logits = _model_forward(
                self.graphdef, lora_params, non_lora_params, input_ids, attention_mask, adapter_indices
            )  # [B, T, V]

            logprobs = jax.nn.log_softmax(logits, axis=-1)  # [B, T, V]
            target_logprobs = jnp.take_along_axis(logprobs, target_ids[..., None], axis=-1).squeeze(-1)

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

            per_seq_loss = per_token_losses.sum(axis=-1) / loss_mask.sum(axis=-1)
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

            replicated = jax.NamedSharding(self.mesh, jax.P(None))

            # JIT the fused function
            self._forward_backward_and_accumulate = jax.jit(
                forward_backward_and_accumulate,
                in_shardings=(accumulated_grads_shardings, lora_shardings, non_lora_shardings) + (replicated,) * 8,
                out_shardings=(accumulated_grads_shardings, replicated, replicated),
                donate_argnames=("accumulated_grads",),
            )
            self._forward = jax.jit(
                forward_only,
                in_shardings=(accumulated_grads_shardings, lora_shardings, non_lora_shardings) + (replicated,) * 8,
                out_shardings=(accumulated_grads_shardings, replicated, replicated),
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

    def _filter_valid_requests(
        self,
        requests: dict[str, tuple[str, any]],
    ) -> tuple[dict[str, any], dict[str, tuple[str, any]]]:
        """Filter out requests with invalid model_ids and return error results for them.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples

        Returns:
            Tuple of (error_results, valid_requests)
        """
        results = {}
        valid_requests = {}

        for request_id, (model_id, request_data) in requests.items():
            if model_id and model_id not in self.models:
                results[request_id] = types.ErrorResponse(error=f"Model {model_id} not loaded", status="failed")
            else:
                valid_requests[request_id] = (model_id, request_data)

        return results, valid_requests

    def find_batchable_model_passes(
        self, session: Session, request_type: types.RequestType
    ) -> dict[str, tuple[str, types.ForwardBackwardInput]]:
        """Find all requests of the given type that come before any destructive update for their model.

        Uses look-ahead scheduling: for each model, only returns operations
        that have no optim_step or load_weights blocking them in the queue.

        Args:
            session: Database session
            request_type: The type of request to find (e.g., FORWARD or FORWARD_BACKWARD)

        Returns:
            Dict mapping request_id to (model_id, request_data) tuples
        """
        # Find the earliest pending optim_step or load_weights per model (these act as barriers)
        barriers_query = (
            select(FutureDB.model_id, func.min(FutureDB.request_id).label("barrier_id"))
            .where(
                (FutureDB.request_type == types.RequestType.OPTIM_STEP)
                | (FutureDB.request_type == types.RequestType.LOAD_WEIGHTS)
            )
            .where(FutureDB.status == RequestStatus.PENDING)
            .group_by(FutureDB.model_id)
        )
        barriers = dict(session.exec(barriers_query).all())

        # Get all pending operations of the requested type ordered by request_id
        query = (
            select(FutureDB)
            .where(FutureDB.request_type == request_type)
            .where(FutureDB.status == RequestStatus.PENDING)
            .order_by(FutureDB.request_id)
        )
        ops = session.exec(query).all()

        # Filter: only include ops that come before their model's barrier
        batchable = [op for op in ops if op.model_id not in barriers or op.request_id < barriers[op.model_id]]

        return {
            f.request_id: (f.model_id, types.ForwardBackwardInput.model_validate(f.request_data)) for f in batchable
        }

    def find_batchable_sample(self, session: Session) -> dict[str, tuple[str, types.SampleInput]]:
        """Find all sample ops that can be safely batched together.

        Returns sample operations ensuring that each model_id has only one checkpoint_id
        to avoid loading different checkpoints for the same model in a single batch.

        If sample_max_num_sequences is configured, limits to that many requests so we don't
        produce partial batches in process_sample_batch. If num_samples > 1 for some requests,
        this may not be perfect, but it's good until we implement continuous batching.

        Args:
            session: Database session

        Returns:
            Dict mapping request_id to (model_id, request_data) tuples
        """
        sample_query = (
            select(FutureDB)
            .where(FutureDB.request_type == types.RequestType.SAMPLE)
            .where(FutureDB.status == RequestStatus.PENDING)
            .order_by(FutureDB.request_id)
        )
        sample_ops = session.exec(sample_query).all()

        batchable = []
        model_checkpoints = {}  # Map from model_id to checkpoint_id of first request to that model
        for op in sample_ops:
            checkpoint_id = op.request_data["checkpoint_id"]
            # Base model requests (empty checkpoint_id) are always compatible, otherwise only
            # take only requests with one checkpoint_id for a given model_id
            if checkpoint_id == "" or model_checkpoints.setdefault(op.model_id, checkpoint_id) == checkpoint_id:
                batchable.append(op)

        if self.config.sample_max_num_sequences > 0:
            batchable = batchable[: self.config.sample_max_num_sequences]

        return {f.request_id: (f.model_id, types.SampleInput.model_validate(f.request_data)) for f in batchable}

    def find_single_requests(self, session: Session) -> dict[str, tuple[str, types.RequestType, dict]]:
        """Find all requests that need to be processed individually (not batchable).

        Args:
            session: Database session

        Returns:
            Dict mapping request_id to (model_id, request_type, request_data) tuples
        """
        statement = (
            select(FutureDB)
            .where(FutureDB.status == RequestStatus.PENDING)
            .where(FutureDB.request_type != types.RequestType.FORWARD_BACKWARD)
            .where(FutureDB.request_type != types.RequestType.FORWARD)
            .where(FutureDB.request_type != types.RequestType.SAMPLE)
            .where(FutureDB.request_type != types.RequestType.EXTERNAL)
            .order_by(FutureDB.request_id)
        )
        other_futures = session.exec(statement).all()

        return {f.request_id: (f.model_id, f.request_type, f.request_data) for f in other_futures}

    def process_create_model(self, model_id: str, request_data: types.CreateModelInput) -> types.CreateModelOutput:
        """Create and initialize a model."""
        # Assign adapter index for this model_id
        adapter_index = max((m.adapter_index for m in self.models.values()), default=0) + 1

        if adapter_index >= self.config.max_lora_adapters:
            raise ValueError(f"Maximum number of LoRA adapters ({self.config.max_lora_adapters}) reached")

        # Extract LoRA configuration
        lora_config = request_data.lora_config

        # Validate rank doesn't exceed max
        if not (0 < lora_config.rank <= self.config.max_lora_rank):
            raise ValueError(f"LoRA rank {lora_config.rank} must be between 1 and {self.config.max_lora_rank}")

        self.models[model_id] = types.ModelMetadata(
            adapter_index=adapter_index,
            lora_config=lora_config,
        )

        with jax.set_mesh(self.mesh):
            # These values are always overridden by the hyperparams in the optim_step request.
            tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
            self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.model.is_lora_param)

        # Update the adapter's rank and scaling in all LoRA layers
        update_adapter_config(self.model, adapter_index, lora_config)

        logger.info(f"Created LoRA model {model_id} with adapter index {adapter_index}, config {lora_config}")

        return types.CreateModelOutput(
            model_id=model_id,
            base_model=self.config.base_model,
            lora_config=request_data.lora_config,
        )

    def _process_model_pass_batch(
        self,
        requests: dict[str, tuple[str, types.ForwardBackwardInput]],
        model_pass_fn: Callable,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Common batch processing logic for forward-only and forward-backward operations.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples
            model_pass_fn: Callable to perform the model pass (forward or forward_backward)

        Returns:
            Dict mapping request_id to result_data or error info
        """
        results, valid_requests = self._filter_valid_requests(requests)

        if not valid_requests:
            return results

        # Collect all examples and their metadata
        all_input_ids = []
        all_targets = []
        all_token_weights = []
        all_adapter_indices = []
        example_model_ids = []  # map each example to its model_id
        request_batch_slices = []  # Track which examples belong to which request
        all_sampling_logprobs = []
        all_advantages = []
        all_loss_fn_types = []

        for request_id, (model_id, request_data) in valid_requests.items():
            adapter_index = self.models[model_id].adapter_index
            loss_fn_type = LOSS_TYPES[request_data.loss_fn]

            request_start = len(all_input_ids)
            for item in request_data.data:
                tokens = [t for chunk in item.model_input.chunks for t in chunk.tokens]
                all_input_ids.append(tokens)
                loss_fn_inputs = item.loss_fn_inputs
                all_targets.append(loss_fn_inputs.target_tokens.data)
                all_token_weights.append(loss_fn_inputs.weights.data)
                all_sampling_logprobs.append(loss_fn_inputs.logprobs.data)
                all_advantages.append(loss_fn_inputs.advantages.data)
                all_adapter_indices.append(adapter_index)
                example_model_ids.append(model_id)
                all_loss_fn_types.append(loss_fn_type)

            request_batch_slices.append((request_id, model_id, request_start, len(all_input_ids)))

        # Pad sequences to same length. Also bin it so the JIT has to compile fewer kernels.
        max_len = round_up_seq_len(max(len(seq) for seq in all_input_ids))

        input_ids = pad_batch(all_input_ids, max_len, np.int32)
        target_ids = pad_batch(all_targets, max_len, np.int32)
        adapter_indices = jnp.array(all_adapter_indices, dtype=jnp.int32)
        loss_fn_types = jnp.array(all_loss_fn_types, dtype=jnp.int32)

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

        with jax.set_mesh(self.mesh), self._jit_timing_context(seq_len, mode="train"):
            for mb_start in range(0, total_bs, micro_bs):
                mb_end = min(mb_start + micro_bs, total_bs)
                self.accumulated_grads, per_token_losses, target_logprobs = model_pass_fn(
                    self.accumulated_grads,
                    self.lora_params,
                    self.non_lora_params,
                    input_ids[mb_start:mb_end],
                    attention_mask[mb_start:mb_end],
                    adapter_indices[mb_start:mb_end],
                    target_ids[mb_start:mb_end],
                    loss_mask[mb_start:mb_end],
                    loss_fn_types[mb_start:mb_end],
                    sampling_logprobs[mb_start:mb_end],
                    advantages[mb_start:mb_end],
                )
                token_losses_device.append(per_token_losses)
                logprobs_device.append(target_logprobs)

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

    def process_forward_backward_batch(self, requests):
        return self._process_model_pass_batch(requests, self._forward_backward_and_accumulate)

    def process_forward_batch(self, requests):
        return self._process_model_pass_batch(requests, self._forward)

    def process_sample_batch(
        self, requests: dict[str, tuple[str, types.SampleInput]]
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Process multiple sample requests in a single batch

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples

        Returns:
            Dict mapping request_id --> result_data or error info
        """
        results, valid_requests = self._filter_valid_requests(requests)

        if not valid_requests:
            return results

        # Computes prompt_logprobs for the whole batch if any request asked for them
        needs_prompt_logprobs = any(request_data.prompt_logprobs for (_, request_data) in valid_requests.values())

        all_prompts = []
        all_sampling_params = []
        all_adapter_indices = []
        request_batch_slices = []

        adapter_indices_batch = self.load_sampler_weights(valid_requests)

        for i, (request_id, (model_id, request_data)) in enumerate(valid_requests.items()):
            request_start = len(all_prompts)

            # Expand requests for num_samples (TODO: Once we have continuous batching /
            # paged attention, we should do the prefill only once and share the kv cache)
            for _ in range(request_data.num_samples):
                prompt_tokens = [token for chunk in request_data.prompt.chunks for token in chunk.tokens]
                all_prompts.append(prompt_tokens)
                all_sampling_params.append(request_data.sampling_params)
                all_adapter_indices.append(adapter_indices_batch[i])

            request_batch_slices.append((request_id, model_id, request_start, len(all_prompts), request_data))

        total_batch_size = len(all_prompts)
        max_batch_size = (
            self.config.sample_max_num_sequences if self.config.sample_max_num_sequences > 0 else total_batch_size
        )
        # Collect generated sequences and prompt logprobs across batches
        all_sequences: list[types.GeneratedSequence] = []
        all_prompt_logprobs: list[list[float]] = []

        with jax.set_mesh(self.mesh):
            model = nnx.merge(self.graphdef, self.lora_params, self.non_lora_params)
            for batch_start in range(0, total_batch_size, max_batch_size):
                batch_end = min(batch_start + max_batch_size, total_batch_size)
                batch_prompts = pad(all_prompts[batch_start:batch_end], max_batch_size, fill=[])
                adapter_indices = pad(all_adapter_indices[batch_start:batch_end], max_batch_size, fill=0)
                sampling_params = pad(
                    all_sampling_params[batch_start:batch_end], max_batch_size, fill=all_sampling_params[batch_start]
                )

                # Pad sequences to same length within the batch to minimize memory usage.
                # Also bin it so the JIT has to compile fewer kernels.
                # Use left-padding for sampling so the last position is always the last real token.
                max_len = round_up_seq_len(max((len(seq) for seq in batch_prompts), default=0))
                input_ids = pad_batch(batch_prompts, max_len, np.int32, left=True)
                attention_mask = pad_batch([[1] * len(seq) for seq in batch_prompts], max_len, np.int32, left=True)

                with self._jit_timing_context(max_len, mode="sample"):
                    result = model.generate(
                        input_ids,
                        attention_mask,
                        sampling_params=sampling_params,
                        adapter_indices=jnp.array(adapter_indices, dtype=jnp.int32),
                        prompt_logprobs=needs_prompt_logprobs,
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

        for request_id, _, start_idx, end_idx, request_data in request_batch_slices:
            sequences = [all_sequences[i] for i in range(start_idx, end_idx)]
            # Each of `num_samples` samples in a request share the same prompt; use the first's prompt logprobs
            prompt_logprobs = (
                all_prompt_logprobs[start_idx] if request_data.prompt_logprobs and all_prompt_logprobs else None
            )
            results[request_id] = types.SampleOutput(sequences=sequences, prompt_logprobs=prompt_logprobs)

        return results

    def process_optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        """Process an optim_step request and apply accumulated gradients."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        adapter_index = jnp.int32(self.models[model_id].adapter_index)

        # Check if we have any gradients accumulated (count > 0)
        if self.accumulated_grads.counts[adapter_index] == 0:
            logger.warning(f"No accumulated gradients for model {model_id}, skipping optimizer step")
            return types.OptimStepOutput()

        # Update hyperparameters from the request
        hp = self.optimizers[model_id].opt_state.hyperparams
        hp["learning_rate"][...] = request_data.adam_params.learning_rate
        hp["b1"][...] = request_data.adam_params.beta1
        hp["b2"][...] = request_data.adam_params.beta2
        hp["eps"][...] = request_data.adam_params.eps

        # JIT-compiled: compute full gradients, apply optimizer update, and reset accumulated grads
        with jax.set_mesh(self.mesh):
            self.accumulated_grads = self._compute_grads_and_update(
                self.accumulated_grads,
                self.lora_params,
                self.optimizers[model_id],
                adapter_index,
            )

        logger.info(f"Applied optimizer step for model {model_id} (adapter {adapter_index})")
        return types.OptimStepOutput()

    def process_load_weights(self, model_id: str, request_data: types.LoadWeightsInput) -> types.LoadWeightsOutput:
        """Loads a clean, trimmed training checkpoint."""
        if model_id not in self.models:
            raise ValueError("Model not loaded. Create the model before loading a checkpoint.")

        adapter_index = self.models[model_id].adapter_index
        checkpoint_dir = (
            self.config.checkpoints_base / request_data.source_model_id / f"{request_data.checkpoint_id}.tar.gz"
        )

        with download_and_unpack(checkpoint_dir) as temp_dir:
            checkpoint = checkpoints.restore_checkpoint(
                ckpt_dir=temp_dir, target=self._extract_checkpoint_data(model_id), prefix="checkpoint_"
            )

        if checkpoint is None:
            raise FileNotFoundError(f"Training checkpoint not found in {checkpoint_dir}")

        # Validate rank
        rank = checkpoint["lora_config"]["rank"]
        if self.models[model_id].lora_config.rank != rank:
            raise ValueError(
                f"Rank mismatch: checkpoint has rank {rank}, model configured with rank {self.models[model_id].lora_config.rank}"
            )

        # Update both LoRA weights and optimizer state
        insert_adapter_state(adapter_index, self.lora_params, checkpoint["lora_weights"], rank)
        insert_adapter_state(adapter_index, nnx.state(self.optimizers[model_id]), checkpoint["optimizer_state"], rank)

        logger.info(f"Loaded training checkpoint for model {model_id} from {checkpoint_dir}")
        return types.LoadWeightsOutput(type="load_weights")

    def process_save_weights(self, model_id: str, request_data: types.SaveWeightsInput) -> types.SaveWeightsOutput:
        """
        Saves a clean training checkpoint by converting the trimmed NNX graph
        to a pure dictionary before serialization, following official Flax docs.
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        checkpoint_id = request_data.path
        output_path = self.config.checkpoints_base / model_id / f"{checkpoint_id}.tar.gz"

        with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.TRAINING):
            with pack_and_upload(output_path) as temp_dir:
                checkpoints.save_checkpoint(
                    target=self._extract_checkpoint_data(model_id),
                    ckpt_dir=temp_dir,
                    step=0,
                    prefix="checkpoint_",
                    overwrite=True,
                )

            logger.info(f"Saved trimmed training checkpoint for model {model_id} to {output_path}")

        return types.SaveWeightsOutput(
            path=f"tinker://{model_id}/weights/{checkpoint_id}",
            type="save_weights",
        )

    def process_save_weights_for_sampler(
        self, model_id: str, request_data: types.SaveWeightsForSamplerInput
    ) -> types.SaveWeightsForSamplerOutput:
        """Process a save_weights_for_sampler request and save model weights."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        lora_model = self.models[model_id]

        # Make sure the user cannot store checkpoints in places like ../../<important file>
        checkpoint_id = Path(request_data.path).name
        output_path = self.config.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"

        with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.SAMPLER):
            # Save the LoRA adapter weights and LoRA config as tar.gz
            save_lora_checkpoint(
                self.model, self.config.base_model, lora_model.lora_config, lora_model.adapter_index, output_path
            )

            logger.info(
                f"Saved LoRA adapter weights for model {model_id} (adapter {lora_model.adapter_index}) to {output_path}"
            )

        return types.SaveWeightsForSamplerOutput(
            path=f"tinker://{model_id}/{checkpoint_id}",
            type="save_weights_for_sampler",
        )

    def load_sampler_weights(self, requests: dict[str, tuple[str, types.SampleInput]]) -> list[int]:
        """Load sampler weights for all requests and return full adapter indices array.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples for the batch

        Returns:
            The adapter_indices array for LoRA sampling [batch_size]
            Uses adapter index 0 for base model sampling (no LoRA)
        """
        adapter_indices = []

        for _, (model_id, request_data) in requests.items():
            base_model = request_data.base_model
            checkpoint_id = request_data.checkpoint_id
            if base_model is None:
                # This code path is for sampling from a LoRA adapter
                assert checkpoint_id != "", "checkpoint_id must be not empty"

                adapter_index = self.models[model_id].adapter_index
                if self.models[model_id].loaded_checkpoint_id == checkpoint_id:
                    # Load model from RAM
                    adapter_indices.append(adapter_index)
                else:
                    # Load model from disk
                    assert adapter_index not in adapter_indices, "Cannot override already used adapter"

                    checkpoint_path = (
                        self.config.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"
                    )
                    logger.info(f"Loading LoRA sampler checkpoint from {checkpoint_path}")
                    adapter_config = self.models[model_id].lora_config
                    load_lora_checkpoint(self.model, adapter_config, adapter_index, checkpoint_path)

                    self.models[model_id].loaded_checkpoint_id = checkpoint_id
                    logger.info(f"Loaded LoRA sampler weights for model {model_id} at adapter index {adapter_index}")
                    adapter_indices.append(adapter_index)
            else:
                # This code path is for sampling from the base model
                if base_model != self.config.base_model:
                    raise ValueError(
                        f"Requested base_model '{base_model}' does not match engine's base_model '{self.config.base_model}'"
                    )
                assert model_id == "" and checkpoint_id == ""
                adapter_indices.append(0)

        return adapter_indices

    def _complete_futures(self, results: dict[str, BaseModel]):
        """Helper method to complete multiple futures in the database.

        Args:
            results: Dict mapping request_id to result (Pydantic BaseModel)
        """
        completed_at = datetime.now(timezone.utc)
        params = [
            {
                "request_id": int(request_id),
                "result_data": result.model_dump(),
                "status": RequestStatus.FAILED if isinstance(result, types.ErrorResponse) else RequestStatus.COMPLETED,
                "completed_at": completed_at,
            }
            for request_id, result in results.items()
        ]

        with Session(self.db_engine) as session:
            session.execute(update(FutureDB), params)
            session.commit()

    def process_single_request(self, request_type: types.RequestType, model_id: str, request_data: dict) -> BaseModel:
        match request_type:
            case types.RequestType.CREATE_MODEL:
                return self.process_create_model(model_id, types.CreateModelInput.model_validate(request_data))
            case types.RequestType.OPTIM_STEP:
                return self.process_optim_step(model_id, types.OptimStepInput.model_validate(request_data))
            case types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER:
                return self.process_save_weights_for_sampler(
                    model_id, types.SaveWeightsForSamplerInput.model_validate(request_data)
                )
            case types.RequestType.SAVE_WEIGHTS:
                return self.process_save_weights(model_id, types.SaveWeightsInput.model_validate(request_data))
            case types.RequestType.LOAD_WEIGHTS:
                return self.process_load_weights(model_id, types.LoadWeightsInput.model_validate(request_data))
            case _:
                raise ValueError(f"Unknown request type: {request_type}")

    def process_single_requests(self, requests: dict[str, tuple[str, types.RequestType, dict]]):
        """Process a collection of single (non-batchable) requests.

        Args:
            requests: Dict mapping request_id to (model_id, request_type, request_data) tuples
        """
        if not requests:
            return
        results = {}
        for request_id, (model_id, request_type, request_data) in requests.items():
            with log_timing(f"process_single_request({request_type.value})"):
                try:
                    result = self.process_single_request(request_type, model_id, request_data)
                except Exception as e:
                    logger.exception(f"Error processing request {request_id}: {e}")
                    result = types.ErrorResponse(error=str(e), status="failed")
            results[request_id] = result
        self._complete_futures(results)

    def process_batch_requests(self, requests: dict[str, tuple[str, BaseModel]], batch_processor):
        """Generic function to process a batch of requests.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples
            batch_processor: Function to call to process the batch
        """
        if not requests:
            return
        with log_timing(f"process_batch_requests({batch_processor.__name__}, n={len(requests)})"):
            try:
                results = batch_processor(requests)
            except Exception as e:
                logger.exception(f"Error processing batch: {e}")
                results = {request_id: types.ErrorResponse(error=str(e), status="failed") for request_id in requests}
        self._complete_futures(results)

    def process_pending_requests(self):
        """Main loop to process pending requests."""
        while True:
            # Query for pending requests and extract data within session context
            with Session(self.db_engine) as session:
                # Use look-ahead scheduling to find batchable forward_backward and forward model passes
                forward_backward_requests = self.find_batchable_model_passes(
                    session, types.RequestType.FORWARD_BACKWARD
                )
                forward_requests = self.find_batchable_model_passes(session, types.RequestType.FORWARD)
                # Find pending sample requests that can be batched
                sample_requests = self.find_batchable_sample(session)
                # Get other pending requests (non forward_backward and non sampling)
                other_requests = self.find_single_requests(session)

            # Process batches outside of session context
            self.process_batch_requests(forward_backward_requests, self.process_forward_backward_batch)
            self.process_batch_requests(forward_requests, self.process_forward_batch)
            self.process_batch_requests(sample_requests, self.process_sample_batch)

            # Process other request types individually (in the future we can also batch independent optim_steps)
            self.process_single_requests(other_requests)

            # Poll every 100ms
            time.sleep(0.1)

    def run(self):
        """Entry point to start the engine."""
        logger.info("Starting background engine...")
        self.process_pending_requests()


def main():
    """Entry point for the background engine."""
    # Create argument parser and add Pydantic model fields
    parser = argparse.ArgumentParser(description="SkyRL tx tinker engine for processing requests")
    add_model(parser, EngineConfig)

    # Parse command-line arguments
    args = parser.parse_args()

    # Create EngineConfig from parsed arguments
    config = EngineConfig.model_validate(vars(args))

    # Initialize and run the engine
    TinkerEngine(config).run()


if __name__ == "__main__":
    main()
