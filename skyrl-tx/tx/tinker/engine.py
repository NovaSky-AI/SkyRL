"""Background engine for processing training requests."""

import argparse
import functools
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel
from sqlmodel import create_engine, Session, select, func

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import checkpoints


import optax
from transformers import AutoConfig
from huggingface_hub import snapshot_download

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
)
from tx.layers.lora import update_adapter_config
from tx.utils.log import logger


@dataclass
class AccumulatedGradients:
    """Stores accumulated gradients for a LoRA adapter."""

    grad_sum: nnx.State | None = None
    denominator: int = 0

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("adapter_index",))
    def _accumulate(grad_sum: nnx.State, lora_grads: nnx.State, adapter_index: int) -> nnx.State:
        """Extracts gradients and adds them to the sum."""
        return jax.tree.map(lambda accum, g: accum + g[adapter_index], grad_sum, lora_grads)

    def add(self, lora_grads: nnx.State, adapter_index: int, count: int) -> None:
        """Accumulate gradients and increment denominator."""
        if self.grad_sum is None:
            self.grad_sum = jax.tree.map(lambda g: g[adapter_index], lora_grads)
        else:
            self.grad_sum = self._accumulate(self.grad_sum, lora_grads, adapter_index)
        self.denominator += count

    def get_mean(self) -> nnx.State:
        """Compute mean gradients."""
        if self.grad_sum is None or self.denominator == 0:
            raise ValueError("Cannot compute mean: no gradients accumulated")
        return jax.tree.map(
            lambda g: g / jnp.asarray(self.denominator, dtype=g.dtype),
            self.grad_sum,
        )

    def reset(self) -> None:
        """Clear accumulated gradients."""
        self.grad_sum = None
        self.denominator = 0


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
        # Store accumulated gradients per LoRA adapter (model_id -> accumulated gradients)
        self.accumulated_grads: dict[str, AccumulatedGradients] = {}
        # Store optimizer instances per LoRA adapter (model_id -> optimizer)
        self.optimizers: dict[str, nnx.Optimizer] = {}
        # Metrics recorded in the engine
        self.metrics = types.EngineMetrics()

        # Initialize the shared base model
        self.model_config = AutoConfig.from_pretrained(self.config.base_model)

        # Configure LoRA settings
        self.model_config.max_lora_adapters = self.config.max_lora_adapters
        self.model_config.max_lora_rank = self.config.max_lora_rank
        self.model_config.shard_attention_heads = self.config.shard_attention_heads

        model_class = get_model_class(self.model_config)

        # Download model weights from HuggingFace
        checkpoint_path = snapshot_download(self.config.base_model, allow_patterns=["*.safetensors"])

        # Create model and load weights
        self.mesh = jax.make_mesh((1, self.config.tensor_parallel_size), ("dp", "tp"))
        with jax.set_mesh(self.mesh):
            self.model = model_class(self.model_config, dtype=get_dtype(self.model_config.dtype), rngs=nnx.Rngs(0))
            load_safetensors(checkpoint_path, self.model_config, self.model)

            # Split model into LoRA and non-LoRA parameters
            self.graphdef, self.lora_params, self.non_lora_params = nnx.split(self.model, self.model.is_lora_param, ...)
            update_adapter_config(self.model, adapter_index=0, lora_rank=1, lora_alpha=1.0)

        logger.info(
            f"Initialized base model {self.config.base_model} with max_lora_adapters={self.config.max_lora_adapters}, max_lora_rank={self.config.max_lora_rank}"
        )

        self._create_loss_and_grad_fn()

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
            model: nnx.Module, input_ids: jax.Array, attention_mask: jax.Array, adapter_indices: jax.Array
        ) -> jax.Array:
            output = model(input_ids, attention_mask=attention_mask, adapter_indices=adapter_indices)
            return output.logits

        if self.config.gradient_checkpointing:
            # policy=None corresponds full activation recomputation
            _model_forward = nnx.remat(_model_forward, policy=None)

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
            model = nnx.merge(self.graphdef, lora_params, non_lora_params)
            logits = _model_forward(model, input_ids, attention_mask, adapter_indices)  # [B, T, V]

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
            return per_seq_loss.sum(), (logits, per_token_losses)

        # Only differentiate with respect to lora_params (argnums=0)
        loss_and_grad_fn = jax.value_and_grad(loss_for_lora, argnums=0, has_aux=True)

        if self.config.enforce_eager:
            # Disable JIT compilation for debugging
            self._loss_and_grad_fn = loss_and_grad_fn
        else:
            # Retrieve the sharding of lora and non_lora params and compute the sharding of inputs and outputs
            lora_shardings = jax.tree.map(
                lambda spec: jax.NamedSharding(self.mesh, spec), nnx.get_partition_spec(self.lora_params)
            )
            non_lora_shardings = jax.tree.map(
                lambda spec: jax.NamedSharding(self.mesh, spec), nnx.get_partition_spec(self.non_lora_params)
            )
            replicated = jax.NamedSharding(self.mesh, jax.P(None))
            scalar = jax.NamedSharding(self.mesh, jax.P())
            self._loss_and_grad_fn = jax.jit(
                loss_and_grad_fn,
                # One input sharding parameter for each argument of loss_for_lora
                in_shardings=(lora_shardings, non_lora_shardings) + (replicated,) * 8,
                # One output sharding parameter for each return value of loss_for_lora
                out_shardings=((scalar, (replicated, replicated)), lora_shardings),
            )

    def _micro_batch_size(self, total: int) -> int:
        """Return effective micro-batch size; 0/absent => disabled (use full fused batch)."""
        mb = self.config.train_micro_batch_size
        return total if mb <= 0 else max(1, min(mb, total))

    @contextmanager
    def _jit_timing_context(self, seq_len: int):
        """Context manager to track JIT compilation times for different sequence lengths."""
        if not self.config.enforce_eager and seq_len not in self.metrics.seq_len_jit_times:
            logger.info(f"JIT compiling for seq_len={seq_len} in progress...")
            start_time = time.time()
            yield
            elapsed = time.time() - start_time
            self.metrics.seq_len_jit_times[seq_len] = elapsed
            logger.info(f"JIT compilation for seq_len={seq_len} took {elapsed:.2f}s")
        else:
            yield

    def _forward_backward(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        adapter_indices: jax.Array,
        target_ids: jax.Array,
        loss_mask: jax.Array,
        loss_fn_types: jax.Array,
        sampling_logprobs: jax.Array,
        advantages: jax.Array,
    ) -> tuple[jax.Array, jax.Array, nnx.State]:
        """Run forward+backward on a batch of inputs."""
        seq_len = input_ids.shape[1]
        with jax.set_mesh(self.mesh), self._jit_timing_context(seq_len):
            (_, (logits, per_token_losses)), lora_grads = self._loss_and_grad_fn(
                self.lora_params,
                self.non_lora_params,
                input_ids,
                attention_mask,
                adapter_indices,
                target_ids,
                loss_mask,
                loss_fn_types,
                sampling_logprobs,
                advantages,
            )
        logprobs = jax.nn.log_softmax(logits, axis=-1)  # [B, T, V]
        target_logprobs = jnp.take_along_axis(logprobs, target_ids[..., None], axis=-1).squeeze(-1)  # [B, T]
        return per_token_losses, target_logprobs, lora_grads

    def _accumulate_grads(self, lora_grads: nnx.State, example_model_ids: list[str]) -> None:
        """
        Accumulate adapter-wise gradient sums and example counts.
        """
        for model_id, count in Counter(example_model_ids).items():
            adapter_index = self.models[model_id].adapter_index
            accumulator = self.accumulated_grads[model_id]
            # Extract and accumulate gradients for this adapter
            accumulator.add(lora_grads, adapter_index, count)

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

    def find_batchable_forward_backward(self, session: Session) -> dict[str, tuple[str, types.ForwardBackwardInput]]:
        """Find all forward_backward ops that come before any destructive update for their model.

        Uses look-ahead scheduling: for each model, only returns forward_backward operations
        that have no optim_step or load_weights blocking them in the queue.

        Args:
            session: Database session

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

        # Get all pending forward_backward operations ordered by request_id
        fwd_bwd_query = (
            select(FutureDB)
            .where(FutureDB.request_type == types.RequestType.FORWARD_BACKWARD)
            .where(FutureDB.status == RequestStatus.PENDING)
            .order_by(FutureDB.request_id)
        )
        fwd_bwd_ops = session.exec(fwd_bwd_query).all()

        # Filter: only include ops that come before their model's barrier
        batchable = [op for op in fwd_bwd_ops if op.model_id not in barriers or op.request_id < barriers[op.model_id]]

        return {
            f.request_id: (f.model_id, types.ForwardBackwardInput.model_validate(f.request_data)) for f in batchable
        }

    def find_batchable_sample(self, session: Session) -> dict[str, tuple[str, types.SampleInput]]:
        """Find all sample ops that can be safely batched together.

        Returns sample operations ensuring that each model_id has only one checkpoint_id
        to avoid loading different checkpoints for the same model in a single batch.

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
            .where(FutureDB.request_type != types.RequestType.SAMPLE)
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

        # Extract LoRA rank and alpha from config
        lora_rank = request_data.lora_config.rank
        lora_alpha = request_data.lora_config.alpha

        # Validate rank doesn't exceed max
        if not (0 < lora_rank <= self.config.max_lora_rank):
            raise ValueError(f"LoRA rank {lora_rank} must be between 1 and {self.config.max_lora_rank}")

        self.models[model_id] = types.ModelMetadata(
            adapter_index=adapter_index,
            lora_config=request_data.lora_config,
        )
        self.accumulated_grads[model_id] = AccumulatedGradients()

        with jax.set_mesh(self.mesh):
            # These values are always overridden by the hyperparams in the optim_step request.
            tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
            self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.model.is_lora_param)

        # Update the adapter's rank and scaling in all LoRA layers
        update_adapter_config(self.model, adapter_index, lora_rank, lora_alpha)

        logger.info(
            f"Created LoRA model {model_id} with adapter index {adapter_index}, rank {lora_rank}, alpha {lora_alpha}"
        )

        return types.CreateModelOutput(
            model_id=model_id,
            base_model=self.config.base_model,
            lora_config=request_data.lora_config,
        )

    def process_forward_backward_batch(
        self, requests: dict[str, tuple[str, types.ForwardBackwardInput]]
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process multiple forward_backward requests in a single batch.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples

        Returns:
            Dict mapping request_id -> result_data or error info
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

        input_ids = jnp.array([seq + [0] * (max_len - len(seq)) for seq in all_input_ids], dtype=jnp.int32)
        target_ids = jnp.array([seq + [0] * (max_len - len(seq)) for seq in all_targets], dtype=jnp.int32)
        adapter_indices = jnp.array(all_adapter_indices, dtype=jnp.int32)
        loss_fn_types = jnp.array(all_loss_fn_types, dtype=jnp.int32)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = jnp.array(
            [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in all_input_ids], dtype=jnp.int32
        )
        loss_mask = jnp.array(
            [all_token_weights[i] + [0] * (max_len - len(all_input_ids[i])) for i in range(len(all_token_weights))],
            dtype=jnp.float32,
        )
        sampling_logprobs = jnp.array(
            [seq + [0.0] * (max_len - len(seq)) for seq in all_sampling_logprobs], dtype=jnp.float32
        )
        advantages = jnp.array([seq + [0.0] * (max_len - len(seq)) for seq in all_advantages], dtype=jnp.float32)

        total_bs = int(input_ids.shape[0])
        micro_bs = self._micro_batch_size(total_bs)
        seq_lens = [len(seq) for seq in all_input_ids]

        # Used to collect per-example outputs (by global row index)
        token_losses_out = [None] * total_bs
        logprobs_out = [None] * total_bs

        for mb_start in range(0, total_bs, micro_bs):
            mb_end = min(mb_start + micro_bs, total_bs)
            per_token_losses, target_logprobs, lora_grads_mb = self._forward_backward(
                input_ids[mb_start:mb_end],
                attention_mask[mb_start:mb_end],
                adapter_indices[mb_start:mb_end],
                target_ids[mb_start:mb_end],
                loss_mask[mb_start:mb_end],
                loss_fn_types[mb_start:mb_end],
                sampling_logprobs[mb_start:mb_end],
                advantages[mb_start:mb_end],
            )
            for i_local, i_global in enumerate(range(mb_start, mb_end)):
                L = seq_lens[i_global]
                token_losses_out[i_global] = per_token_losses[i_local, :L].astype(jnp.float32)
                logprobs_out[i_global] = target_logprobs[i_local, :L].astype(jnp.float32)
            self._accumulate_grads(lora_grads_mb, example_model_ids[mb_start:mb_end])

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

            request_batch_slices.append((request_id, model_id, request_start, len(all_prompts)))

        total_batch_size = len(all_prompts)
        max_batch_size = (
            self.config.sample_max_num_sequences if self.config.sample_max_num_sequences > 0 else total_batch_size
        )

        # Collect generated sequences across batches
        all_sequences: list[types.GeneratedSequence] = []

        with jax.set_mesh(self.mesh):
            model = nnx.merge(self.graphdef, self.lora_params, self.non_lora_params)
            for batch_start in range(0, total_batch_size, max_batch_size):
                batch_end = min(batch_start + max_batch_size, total_batch_size)
                batch_prompts = all_prompts[batch_start:batch_end]

                # Pad sequences to same length within the batch to minimize memory usage.
                max_len = max(len(seq) for seq in batch_prompts) if batch_prompts else 0
                input_ids = jnp.array(
                    [seq + [0] * (max_len - len(seq)) for seq in batch_prompts],
                    dtype=jnp.int32,
                )
                attention_mask = jnp.array(
                    [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in batch_prompts],
                    dtype=jnp.int32,
                )
                adapter_indices = jnp.array(all_adapter_indices[batch_start:batch_end], dtype=jnp.int32)
                sampling_params = all_sampling_params[batch_start:batch_end]

                result = model.generate(
                    input_ids,
                    attention_mask,
                    sampling_params=sampling_params,
                    adapter_indices=adapter_indices,
                )
                all_sequences.extend(
                    types.GeneratedSequence(stop_reason=stop_reason, tokens=tokens, logprobs=logprobs)
                    for stop_reason, tokens, logprobs in zip(result.stop_reasons, result.generated_ids, result.logprobs)
                )

        for request_id, _, start_idx, end_idx in request_batch_slices:
            sequences = [all_sequences[i] for i in range(start_idx, end_idx)]
            results[request_id] = types.SampleOutput(sequences=sequences, prompt_logprobs=[])

        return results

    def process_optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        """Process an optim_step request and apply accumulated gradients."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        adapter_index = self.models[model_id].adapter_index

        # Get accumulated gradients for this adapter
        accumulator = self.accumulated_grads[model_id]
        if accumulator.grad_sum is None or accumulator.denominator == 0:
            logger.warning(f"No accumulated gradients for model {model_id}, skipping optimizer step")
            return types.OptimStepOutput()

        # Average over all examples for this adapter
        adapter_grads = accumulator.get_mean()

        # Create full gradient structure with zeros for all adapters except this one
        def expand_adapter_grads(lora_param, adapter_grad):
            # Create zeros for all adapters with the same shape as lora_param
            full_grads = jnp.zeros_like(lora_param)
            # Set gradients for this specific adapter
            return full_grads.at[adapter_index].set(adapter_grad)

        full_lora_grads = jax.tree.map(expand_adapter_grads, self.lora_params, adapter_grads)

        # Apply optimizer update with hyperparameters from the request
        hp = self.optimizers[model_id].opt_state.hyperparams
        hp["learning_rate"][...] = request_data.adam_params.learning_rate
        hp["b1"][...] = request_data.adam_params.beta1
        hp["b2"][...] = request_data.adam_params.beta2
        hp["eps"][...] = request_data.adam_params.eps
        self.optimizers[model_id].update(self.lora_params, full_lora_grads)

        # Clear accumulated gradients
        self.accumulated_grads[model_id].reset()

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
            restored_data = checkpoints.restore_checkpoint(ckpt_dir=temp_dir, target=None, prefix="checkpoint_")

        if restored_data is None:
            raise FileNotFoundError(f"Training checkpoint not found in {checkpoint_dir}")

        # Validate rank
        rank = restored_data["lora_config"]["rank"]
        if self.models[model_id].lora_config.rank != rank:
            raise ValueError(
                f"Rank mismatch: checkpoint has rank {rank}, model configured with rank {self.models[model_id].lora_config.rank}"
            )

        # Update both LoRA weights and optimizer state
        insert_adapter_state(adapter_index, self.lora_params, self.non_lora_params, restored_data["lora_weights"])
        insert_adapter_state(
            adapter_index, nnx.state(self.optimizers[model_id]), self.non_lora_params, restored_data["optimizer_state"]
        )

        logger.info(f"Loaded training checkpoint for model {model_id} from {checkpoint_dir}")
        return types.LoadWeightsOutput(type="load_weights")

    def process_save_weights(self, model_id: str, request_data: types.SaveWeightsInput) -> types.SaveWeightsOutput:
        """
        Saves a clean training checkpoint by converting the trimmed NNX graph
        to a pure dictionary before serialization, following official Flax docs.
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        adapter_index = self.models[model_id].adapter_index
        checkpoint_id = request_data.path
        output_path = self.config.checkpoints_base / model_id / f"{checkpoint_id}.tar.gz"

        with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.TRAINING):
            adapter_lora_params = extract_adapter_state(adapter_index, self.lora_params, self.non_lora_params)
            optimizer_params = extract_adapter_state(
                adapter_index, nnx.state(self.optimizers[model_id]), self.non_lora_params
            )

            with pack_and_upload(output_path) as temp_dir:
                checkpoints.save_checkpoint(
                    target={
                        "lora_weights": nnx.to_pure_dict(adapter_lora_params),
                        "optimizer_state": nnx.to_pure_dict(optimizer_params),
                        "lora_config": self.models[model_id].lora_config.model_dump(),
                    },
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
        output_path = self.config.checkpoints_base / model_id / f"{checkpoint_id}.tar.gz"

        with self._checkpoint_status_context(model_id, checkpoint_id, types.CheckpointType.SAMPLER):
            # Save the LoRA adapter weights and LoRA config as tar.gz
            save_lora_checkpoint(self.model, lora_model.lora_config, lora_model.adapter_index, output_path)

            logger.info(
                f"Saved LoRA adapter weights for model {model_id} (adapter {lora_model.adapter_index}) to {output_path}"
            )

        return types.SaveWeightsForSamplerOutput(
            path=f"tinker://{model_id}/{checkpoint_id}",
            type="save_weights_for_sampler",
        )

    def load_sampler_weights(self, requests: dict[str, tuple[str, types.SampleInput]]) -> jax.Array:
        """Load sampler weights for all requests and return full adapter indices array.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples for the batch

        Returns:
            The adapter_indices array for LoRA sampling [batch_size]
            Uses adapter index 0 for base model sampling (no LoRA)
        """
        adapter_indices_list = []

        for _, (model_id, request_data) in requests.items():
            if request_data.base_model is None:
                # This code path is for sampling from a LoRA adapter
                assert request_data.checkpoint_id != "", "checkpoint_id must be not empty"

                adapter_index = self.models[model_id].adapter_index
                if self.models[model_id].loaded_checkpoint_id == request_data.checkpoint_id:
                    # Load model from RAM
                    adapter_indices_list.append(adapter_index)
                else:
                    # Load model from disk
                    assert adapter_index not in adapter_indices_list, "Cannot override already used adapter"

                    checkpoint_path = self.config.checkpoints_base / model_id / f"{request_data.checkpoint_id}.tar.gz"
                    logger.info(f"Loading LoRA sampler checkpoint from {checkpoint_path}")
                    load_lora_checkpoint(self.model, adapter_index, checkpoint_path)

                    self.models[model_id].loaded_checkpoint_id = request_data.checkpoint_id
                    logger.info(f"Loaded LoRA sampler weights for model {model_id} at adapter index {adapter_index}")
                    adapter_indices_list.append(adapter_index)
            else:
                # This code path is for sampling from the base model
                if request_data.base_model != self.config.base_model:
                    raise ValueError(
                        f"Requested base_model '{request_data.base_model}' does not match engine's base_model '{self.config.base_model}'"
                    )
                assert model_id == "" and request_data.checkpoint_id == ""
                adapter_indices_list.append(0)

        return jnp.array(adapter_indices_list, dtype=jnp.int32)

    def _complete_futures(self, results: dict[str, BaseModel]):
        """Helper method to complete multiple futures in the database.

        Args:
            results: Dict mapping request_id to result (Pydantic BaseModel)
        """
        with Session(self.db_engine) as session:
            for request_id, result in results.items():
                future = session.get(FutureDB, request_id)
                assert future is not None, f"Future with request_id {request_id} not found in database"

                result_data = result.model_dump()
                future.result_data = result_data
                future.status = (
                    RequestStatus.FAILED if isinstance(result, types.ErrorResponse) else RequestStatus.COMPLETED
                )
                future.completed_at = datetime.now(timezone.utc)
                session.add(future)
                if future.status == RequestStatus.COMPLETED:
                    logger.info(f"Completed {future.request_type} request {request_id}")
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

    def process_batch_requests(self, requests: dict[str, tuple[str, BaseModel]], batch_processor):
        """Generic function to process a batch of requests.

        Args:
            requests: Dict mapping request_id to (model_id, request_data) tuples
            batch_processor: Function to call to process the batch (e.g., process_forward_backward_batch)
        """
        if not requests:
            return
        try:
            results = batch_processor(requests)
            self._complete_futures(results)
        except Exception as e:
            logger.exception(f"Error processing batch: {e}")
            self._complete_futures(
                {request_id: types.ErrorResponse(error=str(e), status="failed") for request_id in requests}
            )

    def process_pending_requests(self):
        """Main loop to process pending requests."""
        while True:
            # Query for pending requests and extract data within session context
            with Session(self.db_engine) as session:
                # Use look-ahead scheduling to find batchable forward_backward operations
                forward_backward_requests = self.find_batchable_forward_backward(session)
                # Find pending sample requests that can be batched
                sample_requests = self.find_batchable_sample(session)
                # Get other pending requests (non forward_backward and non sampling)
                other_requests = self.find_single_requests(session)

            # Process batches outside of session context
            self.process_batch_requests(forward_backward_requests, self.process_forward_backward_batch)
            self.process_batch_requests(sample_requests, self.process_sample_batch)

            # Process other request types individually (in the future we can also batch independent optim_steps)
            other_results = {}
            for request_id, (model_id, request_type, request_data) in other_requests.items():
                try:
                    result = self.process_single_request(request_type, model_id, request_data)
                except Exception as e:
                    logger.exception(f"Error processing request {request_id}: {e}")
                    result = types.ErrorResponse(error=str(e), status="failed")
                other_results[request_id] = result

            self._complete_futures(other_results)

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
