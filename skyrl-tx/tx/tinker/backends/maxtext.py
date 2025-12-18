"""MaxText backend for TinkerEngine."""

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from tx.tinker import types
from tx.tinker.config import EngineConfig
from tx.tinker.backends.backend import AbstractBackend
from tx.tinker.backends.utils import pad_batch
from tx.utils.models import round_up_seq_len, convert_maxtext_lora_to_hf
from tx.utils.storage import pack_and_upload
from tx.utils.log import logger

# MaxText imports
import MaxText
from MaxText import maxtext_utils
from MaxText import model_creation_utils as maxtext_model_creation
from MaxText import sharding as maxtext_sharding
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter


def _get_maxtext_base_config_path() -> str:
    """Get the absolute path to MaxText's base.yml config file."""
    import os
    from importlib.resources import files
    try:
        # Try importlib.resources first (works if MaxText packages configs properly)
        config_path = str(files("MaxText").joinpath("configs", "base.yml"))
        if os.path.exists(config_path):
            return config_path
    except (TypeError, FileNotFoundError):
        pass
    # Fallback: derive from package location
    maxtext_pkg_dir = os.path.dirname(MaxText.__file__)
    maxtext_root = os.path.dirname(os.path.dirname(maxtext_pkg_dir))
    config_path = os.path.join(maxtext_root, "src", "MaxText", "configs", "base.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find MaxText base.yml config. Tried: {config_path}. "
            "Ensure MaxText is installed correctly or set MAXTEXT_CONFIG_PATH environment variable."
        )
    return config_path


def parse_maxtext_config(config_str: str):
    """Parse MaxText config from space-separated key=value string."""
    if not config_str:
        return None
    config_path = _get_maxtext_base_config_path()
    logger.info(f"Using MaxText config: {config_path}")
    argv = ["", config_path] + config_str.split()
    from MaxText import pyconfig as maxtext_pyconfig
    return maxtext_pyconfig.initialize(argv)



def _count_params(pytree) -> int:
    """Count total number of parameters in a pytree."""
    def get_numel(x):
        if hasattr(x, 'shape'):
            return int(np.prod(x.shape))
        return 0
    counts = jax.tree.leaves(jax.tree.map(get_numel, pytree))
    return sum(counts)


@jax.tree_util.register_dataclass
@dataclass
class AccumulatedGradients:
    """Stores accumulated gradients."""

    grad_sum: nnx.State
    count: jax.Array

    @classmethod
    def create(cls, lora_params: nnx.State) -> "AccumulatedGradients":
        """Initialize with zeros."""
        return cls(
            grad_sum=jax.tree.map(jnp.zeros_like, lora_params),
            count=jnp.zeros((1,), dtype=jnp.int32),
        )

    def add(self, lora_grads: nnx.State, batch_size: int) -> "AccumulatedGradients":
        """Accumulate gradients and increment count."""
        return AccumulatedGradients(
            grad_sum=jax.tree.map(lambda a, b: a + b, self.grad_sum, lora_grads),
            count=self.count + batch_size,
        )

    def get_mean(self) -> nnx.State:
        """Compute mean gradients."""
        return jax.tree.map(
            lambda g: g / self.count.astype(g.dtype),
            self.grad_sum,
        )

    def reset(self) -> "AccumulatedGradients":
        """Reset gradients and count."""
        return AccumulatedGradients(
            grad_sum=jax.tree.map(jnp.zeros_like, self.grad_sum),
            count=jnp.zeros((1,), dtype=jnp.int32),
        )


class MaxTextBackend(AbstractBackend):
    """Backend for MaxText models with context parallelism.

    This is a single-adapter backend (max_lora_adapters must be 1).
    """

    def __init__(self, config: EngineConfig, maxtext_config):
        """Initialize MaxText backend."""
        if config.max_lora_adapters != 1:
            raise ValueError(
                f"MaxTextBackend only supports single adapter (max_lora_adapters=1), "
                f"got max_lora_adapters={config.max_lora_adapters}"
            )

        self.config = config
        self.maxtext_config = maxtext_config
        self.metrics = types.EngineMetrics()

        # Create mesh using MaxText's device mesh creation
        devices_array = maxtext_utils.create_device_mesh(maxtext_config)
        self.mesh = jax.sharding.Mesh(devices_array, maxtext_config.mesh_axes)
        logger.info(f"Created MaxText mesh with shape {self.mesh.shape}, axes {self.mesh.axis_names}")

        # Create model using MaxText's model creation
        with jax.set_mesh(self.mesh):
            base_model, _ = maxtext_model_creation.create_nnx_model(maxtext_config, mesh=self.mesh)
            self.model = TunixMaxTextAdapter(base_model=base_model)
            self.model.config = None

        # Extract LoRA params for gradient accumulation
        lora_filter = nnx.All(nnx.Param, nnx.Any(nnx.PathContains("lora_a"), nnx.PathContains("lora_b")))
        self.lora_filter = lora_filter
        self.graphdef, self.lora_params, self.non_lora_params = nnx.split(self.model, lora_filter, ...)

        # Initialize accumulated gradients
        self.accumulated_grads = AccumulatedGradients.create(self.lora_params)
        self._log_accumulated_grads()

        # Per-model optimizer storage (managed internally)
        self.optimizers: dict[str, nnx.Optimizer] = {}

        logger.info(f"Initialized MaxText model with context_parallel_size={maxtext_config.context_parallel_size}")

        self._create_loss_and_grad_fn()

    def _log_accumulated_grads(self):
        """Log accumulated gradient structure."""
        accum_params = _count_params(self.accumulated_grads.grad_sum)
        logger.info(f"[MaxText] Accumulated grads total params: {accum_params / 1e6:.2f}M")
        for path, val in jax.tree_util.tree_leaves_with_path(self.accumulated_grads.grad_sum):
            path_str = "/".join(str(k.key) if hasattr(k, 'key') else str(k) for k in path)
            logger.info(f"  {path_str}: {val.shape}")

    def _create_loss_and_grad_fn(self):
        """Create loss and gradient functions for MaxText model."""

        def loss_for_maxtext_model(
            model,
            input_ids: jax.Array,
            positions: jax.Array,
            target_ids: jax.Array,
            loss_mask: jax.Array,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            """Simple cross-entropy loss for MaxText model."""
            logits, _ = model(input_ids, positions, None, None, False)
            logprobs = jax.nn.log_softmax(logits, axis=-1)
            target_logprobs = jnp.take_along_axis(logprobs, target_ids[..., None], axis=-1).squeeze(-1)
            per_token_losses = -target_logprobs * loss_mask
            per_seq_loss = per_token_losses.sum(axis=-1) / (loss_mask.sum(axis=-1) + 1e-8)
            total_loss = per_seq_loss.sum()
            return total_loss, (target_logprobs, per_token_losses)

        loss_and_grad_fn = nnx.value_and_grad(
            loss_for_maxtext_model,
            argnums=nnx.DiffState(0, self.lora_filter),
            has_aux=True
        )

        def forward_backward_maxtext(
            model, input_ids, positions, target_ids, loss_mask,
        ) -> tuple[jax.Array, jax.Array, jax.Array, nnx.State]:
            """Forward-backward for MaxText model."""
            (loss, (target_logprobs, per_token_losses)), grads = loss_and_grad_fn(
                model, input_ids, positions, target_ids, loss_mask,
            )
            return loss, target_logprobs, per_token_losses, grads

        data_sharding = maxtext_sharding.get_input_data_sharding(self.maxtext_config, self.mesh)

        if self.config.enforce_eager:
            self._forward_backward = forward_backward_maxtext
        else:
            with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.maxtext_config.logical_axis_rules):
                self._forward_backward = jax.jit(
                    forward_backward_maxtext,
                    in_shardings=(None, data_sharding, data_sharding, data_sharding, data_sharding),
                )

        def optim_step(model, optimizer, grads):
            """Apply gradients to optimizer."""
            optimizer.update(model, grads)

        if self.config.enforce_eager:
            self._optim_step = optim_step
        else:
            self._optim_step = nnx.jit(optim_step)

        logger.info("Created MaxText loss and gradient functions")

    def _micro_batch_size(self, total: int) -> int:
        """Return effective micro-batch size."""
        mb = self.config.train_micro_batch_size
        return total if mb <= 0 else max(1, min(mb, total))

    @contextmanager
    def _jit_timing_context(self, seq_len: int, mode: str):
        """Context manager to track JIT compilation times."""
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

    def register_model(self, model_id: str, adapter_index: int, lora_config: types.LoraConfig) -> None:
        """Register a new model with the backend.

        Creates optimizer for the model. MaxText is single-adapter so adapter_index is ignored.
        """
        tx = optax.inject_hyperparams(optax.adamw)(learning_rate=0.0)
        self.optimizers[model_id] = nnx.Optimizer(self.model, tx, wrt=self.lora_filter)
        logger.info(f"Registered model {model_id} with MaxText backend")

    def unregister_model(self, model_id: str, adapter_index: int) -> None:
        """Unregister a model from the backend.

        Removes optimizer and zeros LoRA weights.
        """
        self.optimizers.pop(model_id, None)

        # Zero out all LoRA weights (single adapter)
        zeroed_params = jax.tree.map(jnp.zeros_like, self.lora_params)
        nnx.update(self.lora_params, zeroed_params)
        logger.info(f"Unregistered model {model_id} from MaxText backend")

    def precompile_kernels(self, seq_lens: list[int]):
        """Precompile JIT kernels for specified sequence lengths."""
        if not seq_lens or self.config.enforce_eager:
            return

        logger.info(f"Precompiling JIT kernels for sequence lengths: {seq_lens}")
        micro_bs = max(1, self.config.train_micro_batch_size) if self.config.train_micro_batch_size > 0 else 1

        with jax.set_mesh(self.mesh):
            for seq_len in seq_lens:
                dummy_input_ids = jnp.zeros((micro_bs, seq_len), dtype=jnp.int32)
                dummy_target_ids = jnp.zeros((micro_bs, seq_len), dtype=jnp.int32)
                dummy_loss_mask = jnp.ones((micro_bs, seq_len), dtype=jnp.float32)
                dummy_positions = jnp.broadcast_to(jnp.arange(seq_len), (micro_bs, seq_len))

                data_sharding = maxtext_sharding.get_input_data_sharding(self.maxtext_config, self.mesh)
                dummy_input_ids = jax.device_put(dummy_input_ids, data_sharding)
                dummy_positions = jax.device_put(dummy_positions, data_sharding)
                dummy_target_ids = jax.device_put(dummy_target_ids, data_sharding)
                dummy_loss_mask = jax.device_put(dummy_loss_mask, data_sharding)

                with nn_partitioning.axis_rules(self.maxtext_config.logical_axis_rules):
                    with self._jit_timing_context(seq_len, mode="train"):
                        _, _, _, grads = self._forward_backward(
                            self.model, dummy_input_ids, dummy_positions, dummy_target_ids, dummy_loss_mask,
                        )
                        self.accumulated_grads = self.accumulated_grads.add(grads, micro_bs)

                self.accumulated_grads = AccumulatedGradients.create(self.lora_params)

        logger.info(f"Precompilation complete for {len(seq_lens)} sequence lengths")

    def process_forward_backward_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward_backward requests using MaxText model."""
        all_input_ids = prepared_batch.all_input_ids
        all_targets = prepared_batch.all_targets
        all_token_weights = prepared_batch.all_token_weights
        request_batch_slices = prepared_batch.request_batch_slices

        if not all_input_ids:
            return {}

        results = {}
        max_len = round_up_seq_len(max(len(seq) for seq in all_input_ids), self.config.min_seq_len)
        input_ids = pad_batch(all_input_ids, max_len, np.int32)
        target_ids = pad_batch(all_targets, max_len, np.int32)
        loss_mask = pad_batch(all_token_weights, max_len, np.float32)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
        seq_lens = [len(seq) for seq in all_input_ids]

        data_sharding = maxtext_sharding.get_input_data_sharding(self.maxtext_config, self.mesh)

        token_losses_device = []
        logprobs_device = []
        total_bs = batch_size
        micro_bs = self._micro_batch_size(total_bs)

        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.maxtext_config.logical_axis_rules):
            with self._jit_timing_context(seq_len, mode="train"):
                for mb_start in range(0, total_bs, micro_bs):
                    mb_end = min(mb_start + micro_bs, total_bs)
                    print(f"MaxText forward-backward: batch [{mb_start}:{mb_end}], seq_len={seq_len}", flush=True)
                    tic = time.time()

                    mb_input_ids = jax.device_put(input_ids[mb_start:mb_end], data_sharding)
                    mb_positions = jax.device_put(positions[mb_start:mb_end], data_sharding)
                    mb_target_ids = jax.device_put(target_ids[mb_start:mb_end], data_sharding)
                    mb_loss_mask = jax.device_put(loss_mask[mb_start:mb_end], data_sharding)

                    _, target_logprobs, per_token_losses, grads = self._forward_backward(
                        self.model,
                        mb_input_ids,
                        mb_positions,
                        mb_target_ids,
                        mb_loss_mask,
                    )

                    _ = jax.device_get(target_logprobs)

                    took = time.time() - tic
                    tokens_processed = (mb_end - mb_start) * seq_len
                    tokens_per_sec = tokens_processed / took if took > 0 else float('nan')
                    print(f"Batch [{mb_start}:{mb_end}] forward-backward time: {took:.3f} sec, tokens/sec: {tokens_per_sec:,.1f}", flush=True)

                    micro_batch_size = mb_end - mb_start
                    self.accumulated_grads = self.accumulated_grads.add(grads, micro_batch_size)
                    token_losses_device.append(per_token_losses)
                    logprobs_device.append(target_logprobs)

        token_losses_host, logprobs_host = jax.device_get((token_losses_device, logprobs_device))

        token_losses_out = []
        logprobs_out = []
        idx = 0
        for mb_losses, mb_logprobs in zip(token_losses_host, logprobs_host):
            for i in range(mb_losses.shape[0]):
                token_losses_out.append(mb_losses[i, :seq_lens[idx]].astype(jnp.float32))
                logprobs_out.append(mb_logprobs[i, :seq_lens[idx]].astype(jnp.float32))
                idx += 1

        for request_id, _, start_idx, end_idx in request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                token_losses = token_losses_out[i]
                token_logprobs = logprobs_out[i]
                loss_fn_outputs.append({
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
                })

            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )

        return results

    def process_optim_step(
        self,
        model_id: str,
        adapter_index: int,
        request_data: types.OptimStepInput,
    ) -> types.OptimStepOutput:
        """Process an optim_step request."""
        if self.accumulated_grads.count[0] == 0:
            logger.warning(f"No accumulated gradients for model {model_id}, skipping optimizer step")
            return types.OptimStepOutput()

        optimizer = self.optimizers[model_id]
        hp = optimizer.opt_state.hyperparams
        hp["learning_rate"][...] = request_data.adam_params.learning_rate
        hp["b1"][...] = request_data.adam_params.beta1
        hp["b2"][...] = request_data.adam_params.beta2
        hp["eps"][...] = request_data.adam_params.eps

        mean_grads = self.accumulated_grads.get_mean()

        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.maxtext_config.logical_axis_rules):
            self._optim_step(self.model, optimizer, mean_grads)

        self.accumulated_grads = self.accumulated_grads.reset()
        logger.info(f"Applied MaxText optimizer step for model {model_id}")

        return types.OptimStepOutput()

    def process_forward_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        """Process forward-only requests - not implemented for MaxText."""
        raise NotImplementedError("Forward-only pass not yet implemented for MaxText backend")

    def process_sample_batch(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Process sample requests - not implemented for MaxText."""
        raise NotImplementedError("Sampling not yet implemented for MaxText backend")

    def save_checkpoint(
        self,
        output_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Save training checkpoint in HuggingFace PEFT format as tar.gz."""
        with pack_and_upload(output_path) as temp_dir:
            convert_maxtext_lora_to_hf(
                lora_state=self.lora_params,
                output_path=temp_dir,
                base_model_name=self.config.base_model,
                lora_rank=self.maxtext_config.lora_rank,
                lora_alpha=self.maxtext_config.lora_alpha,
            )
        logger.info(f"Saved MaxText training checkpoint to {output_path}")

    def extract_checkpoint_data(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> dict:
        """Extract LoRA state and optimizer state for checkpointing.

        Creates copies of the arrays to ensure the cached state is independent
        from the live model state (which may be zeroed on eviction).
        """
        # Copy arrays to avoid caching references that get zeroed
        lora_weights_copy = jax.tree.map(jnp.copy, self.lora_params)
        optimizer_state_copy = jax.tree.map(jnp.copy, nnx.state(self.optimizers[model_id]))
        return {
            "lora_weights": lora_weights_copy,
            "optimizer_state": optimizer_state_copy,
            "lora_config": models[model_id].lora_config.model_dump(),
        }

    def insert_checkpoint_data(
        self,
        model_id: str,
        checkpoint_data: dict,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Insert checkpoint data into model state.

        Reshards the cached arrays to match the current model's sharding.
        """
        optimizer = self.optimizers[model_id]

        # Reshard cached weights to match current model sharding
        def reshard_to_match(cached, current):
            """Reshard cached array to match current array's sharding."""
            sharding = current.sharding
            return jax.device_put(cached, sharding)

        resharded_lora = jax.tree.map(
            reshard_to_match, checkpoint_data["lora_weights"], self.lora_params
        )
        resharded_optim = jax.tree.map(
            reshard_to_match, checkpoint_data["optimizer_state"], nnx.state(optimizer)
        )

        # Update model state
        nnx.update(self.lora_params, resharded_lora)
        nnx.update(nnx.state(optimizer), resharded_optim)
        logger.info(f"Restored checkpoint data for model {model_id}")

    def save_sampler_checkpoint(
        self,
        output_path,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Save sampler checkpoint in HuggingFace PEFT format as tar.gz."""
        with pack_and_upload(output_path) as temp_dir:
            convert_maxtext_lora_to_hf(
                lora_state=self.lora_params,
                output_path=temp_dir,
                base_model_name=self.config.base_model,
                lora_rank=self.maxtext_config.lora_rank,
                lora_alpha=self.maxtext_config.lora_alpha,
            )
        logger.info(f"Saved MaxText LoRA sampler checkpoint to {output_path}")

    def extract_sampler_weights(
        self,
        model_id: str,
        models: dict[str, types.ModelMetadata],
    ) -> dict:
        """Extract sampler weights."""
        return {
            "lora_params": self.lora_params,
            "lora_rank": self.maxtext_config.lora_rank,
            "lora_alpha": self.maxtext_config.lora_alpha,
        }

    def insert_sampler_weights(
        self,
        model_id: str,
        checkpoint_id: str,
        weights_data: dict,
        models: dict[str, types.ModelMetadata],
    ) -> None:
        """Insert sampler weights - not implemented for MaxText."""
        raise NotImplementedError("Loading sampler weights not yet implemented for MaxText backend")

