"""Native LoRA backend for TinkerEngine (Qwen3 + LoRA).

This backend implements the full training and inference pipeline for Qwen3 models
with LoRA adapters. It uses jax.value_and_grad for gradient computation and supports
multiple LoRA adapters via the AccumulatedGradients dataclass.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.training import checkpoints
from transformers import AutoTokenizer, PretrainedConfig

from tx.models.configs import Qwen3Config
from tx.layers.lora import update_adapter_config
from tx.tinker import types
from tx.tinker.config import EngineConfig
from tx.tinker.backends.backend import AbstractBackend
from tx.tinker.backends.utils import pad, pad_batch
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
from tx.utils.storage import pack_and_upload
from tx.utils.log import logger


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


class NativeBackend(AbstractBackend):
    """Backend for Qwen3 models with LoRA adapters.

    This backend:
    - Uses jax.value_and_grad for gradient computation
    - Uses 2D mesh (dp, tp)
    - Supports multiple LoRA adapters via AccumulatedGradients with counts array
    - Supports both FORWARD and FORWARD_BACKWARD request types
    """

    def __init__(self, config: EngineConfig):
        """Initialize Native LoRA backend."""
        self.config = config
        self.metrics = types.EngineMetrics()

        # Initialize the shared base model with LoRA config
        checkpoint_path = resolve_model_path(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_config = PretrainedConfig.from_pretrained(checkpoint_path)
        self.model_config = Qwen3Config(
            base_config,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            shard_attention_heads=config.shard_attention_heads,
        )

        model_class = get_model_class(self.model_config)

        # Create model and load weights
        self.mesh = jax.make_mesh((1, config.tensor_parallel_size), ("dp", "tp"))
        with jax.set_mesh(self.mesh):
            self.model = model_class(
                self.model_config, dtype=get_dtype(self.model_config.dtype), rngs=nnx.Rngs(0)
            )
            load_safetensors(checkpoint_path, self.model_config, self.model)

            # Split model into LoRA and non-LoRA parameters
            self.graphdef, self.lora_params, self.non_lora_params = nnx.split(
                self.model, self.model.is_lora_param, ...
            )

            # Initialize adapter 0 with dummy config (required for base model sampling path)
            update_adapter_config(self.model, adapter_index=0, lora_config=types.LoraConfig(rank=1, alpha=1.0))

            # Initialize global accumulated gradients
            self.accumulated_grads = AccumulatedGradients.create(
                self.lora_params, config.max_lora_adapters
            )

        # Per-model optimizer storage (managed internally)
        self.optimizers: dict[str, nnx.Optimizer] = {}

        logger.info(
            f"Initialized base model {config.base_model} with "
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
        jit_times = (
            self.metrics.train_seq_len_jit_times
            if mode == "train"
            else self.metrics.sample_seq_len_jit_times
        )
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

            log_sum_exp = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
            target_logits = jnp.take_along_axis(logits, target_ids[..., None], axis=-1)
            target_logprobs = (target_logits - log_sum_exp).squeeze(-1)

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

    def register_model(self, model_id: str, adapter_index: int, lora_config: types.LoraConfig) -> None:
        """Register a new model with the backend.

        Creates optimizer and configures LoRA adapter.
        """
        # Create optimizer
        with jax.set_mesh(self.mesh):
            tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
            self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.model.is_lora_param)

        # Configure adapter
        update_adapter_config(self.model, adapter_index, lora_config)
        logger.info(f"Registered model {model_id} with adapter_index={adapter_index}")

    def unregister_model(self, model_id: str, adapter_index: int) -> None:
        """Unregister a model from the backend.

        Removes optimizer and resets adapter weights.
        """
        # Remove optimizer
        self.optimizers.pop(model_id, None)

        # Zero out adapter weights
        def zero_adapter_slice(path: tuple, p: jnp.ndarray) -> jnp.ndarray:
            if len(path) >= 2 and path[-2].key in {"lora_A", "lora_B"}:
                return p.at[adapter_index].set(0.0)
            return p

        updated_params = jax.tree.map_with_path(zero_adapter_slice, self.lora_params)
        nnx.update(self.lora_params, updated_params)
        logger.info(f"Unregistered model {model_id} (adapter_index={adapter_index})")

    def _process_model_pass_batch(
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
        all_adapter_indices = prepared_batch.all_adapter_indices
        all_loss_fn_types = prepared_batch.all_loss_fn_types
        request_batch_slices = prepared_batch.request_batch_slices

        # Pad sequences to same length. Also bin it so the JIT has to compile fewer kernels.
        max_len = round_up_seq_len(max(len(seq) for seq in all_input_ids), self.config.min_seq_len)

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

    def process_forward_backward_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward_backward requests in a batch."""
        return self._process_model_pass_batch(prepared_batch, self._forward_backward_and_accumulate)

    def process_forward_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward-only requests in a batch (no gradient computation)."""
        return self._process_model_pass_batch(prepared_batch, self._forward)

    def process_optim_step(
        self,
        model_id: str,
        adapter_index: int,
        request_data: types.OptimStepInput,
    ) -> types.OptimStepOutput:
        """Process an optim_step request and apply accumulated gradients."""
        adapter_index_arr = jnp.int32(adapter_index)
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

        # JIT-compiled: compute full gradients, apply optimizer update, and reset accumulated grads
        with jax.set_mesh(self.mesh):
            self.accumulated_grads = self._compute_grads_and_update(
                self.accumulated_grads,
                self.lora_params,
                optimizer,
                adapter_index_arr,
            )

        logger.info(f"Applied optimizer step for model {model_id} (adapter {adapter_index})")
        return types.OptimStepOutput()

    def process_sample_batch(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Process multiple sample requests in a single batch.

        Args:
            prepared_batch: PreparedSampleBatch with all data extracted from requests

        Returns:
            Dict mapping request_id --> result_data or error info
        """
        if not prepared_batch.all_prompts:
            return {}

        results = {}

        # Extract data from prepared batch
        all_prompts = prepared_batch.all_prompts
        all_sampling_params = prepared_batch.all_sampling_params
        all_adapter_indices = prepared_batch.all_adapter_indices
        request_batch_slices = prepared_batch.request_batch_slices
        needs_prompt_logprobs = prepared_batch.needs_prompt_logprobs

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
                batch_adapter_indices = pad(all_adapter_indices[batch_start:batch_end], max_batch_size, fill=0)
                sampling_params = pad(
                    all_sampling_params[batch_start:batch_end], max_batch_size, fill=all_sampling_params[batch_start]
                )

                # Pad sequences to same length within the batch to minimize memory usage.
                # Also bin it so the JIT has to compile fewer kernels.
                # Use left-padding for sampling so the last position is always the last real token.
                max_len = round_up_seq_len(max((len(seq) for seq in batch_prompts), default=0), self.config.min_seq_len)
                input_ids = pad_batch(batch_prompts, max_len, np.int32, left=True)
                attention_mask = pad_batch([[1] * len(seq) for seq in batch_prompts], max_len, np.int32, left=True)

                with self._jit_timing_context(max_len, mode="sample"):
                    result = model.generate(
                        input_ids,
                        attention_mask,
                        sampling_params=sampling_params,
                        adapter_indices=jnp.array(batch_adapter_indices, dtype=jnp.int32),
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

    def save_checkpoint(
        self,
        output_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Save training checkpoint as tar.gz using Flax checkpoints."""
        with pack_and_upload(output_path) as temp_dir:
            checkpoint_data = self.extract_checkpoint_data(model_id, models)
            checkpoints.save_checkpoint(
                target=checkpoint_data,
                ckpt_dir=temp_dir,
                step=0,
                prefix="checkpoint_",
                overwrite=True,
            )
        logger.info(f"Saved training checkpoint to {output_path}")

    def extract_checkpoint_data(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> dict:
        """Extract adapter state and optimizer state for checkpointing."""
        adapter_index = models[model_id].adapter_index
        rank = models[model_id].lora_config.rank
        lora_weights = extract_adapter_state(adapter_index, self.lora_params, rank)
        optimizer_state = extract_adapter_state(adapter_index, nnx.state(self.optimizers[model_id]), rank)
        return {
            "lora_weights": lora_weights,
            "optimizer_state": optimizer_state,
            "lora_config": models[model_id].lora_config.model_dump(),
        }

    def insert_checkpoint_data(
        self,
        model_id: str,
        checkpoint_data: dict,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Insert checkpoint data into model state."""
        adapter_index = models[model_id].adapter_index
        rank = checkpoint_data["lora_config"]["rank"]

        if models[model_id].lora_config.rank != rank:
            raise ValueError(
                f"Rank mismatch: checkpoint has rank {rank}, "
                f"model configured with rank {models[model_id].lora_config.rank}"
            )

        insert_adapter_state(adapter_index, self.lora_params, checkpoint_data["lora_weights"], rank)
        insert_adapter_state(adapter_index, nnx.state(self.optimizers[model_id]), checkpoint_data["optimizer_state"], rank)

    def save_sampler_checkpoint(
        self,
        output_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Save sampler checkpoint as tar.gz using save_lora_checkpoint."""
        lora_model = models[model_id]
        save_lora_checkpoint(
            self.model,
            self.config.base_model,
            lora_model.lora_config,
            lora_model.adapter_index,
            output_path,
        )
        logger.info(f"Saved LoRA sampler checkpoint to {output_path}")

    def extract_sampler_weights(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> dict:
        """Extract weights for sampler checkpoint.

        Returns data needed for save_lora_checkpoint.
        """
        return {
            "model": self.model,
            "base_model": self.config.base_model,
            "lora_config": models[model_id].lora_config,
            "adapter_index": models[model_id].adapter_index,
        }

    def insert_sampler_weights(
        self,
        model_id: str,
        checkpoint_id: str,
        checkpoint_path,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Insert sampler weights from checkpoint file."""
        adapter_index = models[model_id].adapter_index
        adapter_config = models[model_id].lora_config
        load_lora_checkpoint(self.model, adapter_config, adapter_index, checkpoint_path)
        models[model_id].loaded_checkpoint_id = checkpoint_id
        logger.info(f"Loaded LoRA sampler weights for model {model_id} at adapter index {adapter_index}")
