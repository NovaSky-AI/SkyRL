"""Background engine for processing training requests."""

import argparse
import time
import logging
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from sqlmodel import create_engine, Session, select, func

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from transformers import AutoConfig
from huggingface_hub import snapshot_download

from tx.tinker.db_models import FutureDB, DB_PATH, RequestStatus
from tx.tinker import types
from tx.tinker.config import EngineConfig, add_model
from tx.utils.models import get_dtype, get_model_class, save_checkpoint, load_checkpoint
from tx.layers.lora import update_adapter_config
from peft import LoraConfig

logger = logging.getLogger(__name__)

LEARNING_RATE = 1e-4


def round_up_seq_len(seq_len: int) -> int:
    """
    Rounds a sequence length up to roughly two significant binary digits.
    We do this to pad sequences, so the Jax JIT compiler needs to
    compile fewer different shapes.
    """
    if seq_len <= 32:
        return 32

    # Find the position of the most significant bit.
    msb_pos = seq_len.bit_length() - 1
    # Create a mask for the two most significant bits.
    mask = (1 << msb_pos) | (1 << (msb_pos - 1))
    # Round down to the nearest value with at most two significant bits.
    result = seq_len & mask

    # If we rounded down, round up to the next bucket boundary.
    if result < seq_len:
        result += 1 << (msb_pos - 1)

    return result


@dataclass
class AccumulatedGradients:
    """Stores accumulated gradients for a LoRA adapter."""

    grad_sum: nnx.State | None
    denominator: int

    def add(self, grad: nnx.State, count: int) -> None:
        """Accumulate gradients and increment denominator."""
        if self.grad_sum is None:
            self.grad_sum = grad
            self.denominator = count
        else:
            self.grad_sum = jax.tree.map(lambda a, b: a + b, self.grad_sum, grad)
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
        db_path=DB_PATH,
    ):
        """Initialize the engine with a database connection and base model."""
        self.config = config
        self.db_engine = create_engine(f"sqlite:///{db_path}", echo=False)
        # Store LoRA model metadata (model_id -> metadata)
        self.models: dict[str, types.ModelMetadata] = {}
        # Store accumulated gradients per LoRA adapter (model_id -> accumulated gradients)
        self.accumulated_grads: dict[str, AccumulatedGradients] = {}
        # Metrics recorded in the engine
        self.metrics = types.EngineMetrics()

        # Initialize the shared base model
        self.model_config = AutoConfig.from_pretrained(self.config.base_model)

        # Configure LoRA settings
        self.model_config.max_lora_adapters = self.config.max_lora_adapters
        self.model_config.max_lora_rank = self.config.max_lora_rank

        model_class = get_model_class(self.model_config)

        # Download model weights from HuggingFace
        checkpoint_path = snapshot_download(self.config.base_model, allow_patterns=["*.safetensors"])

        # Create model and load weights
        self.mesh = jax.make_mesh((1, self.config.tensor_parallel_size), ("dp", "tp"))
        with jax.set_mesh(self.mesh):
            self.model = model_class(self.model_config, dtype=get_dtype(self.model_config.dtype), rngs=nnx.Rngs(0))
            load_checkpoint(checkpoint_path, self.model_config, self.model)

            # Create optimizer that only targets LoRA A and B parameters
            def is_lora_param(path, value):
                return any(name in path for name in ["lora_A", "lora_B"])

            self.optimizer = nnx.Optimizer(self.model, optax.adamw(LEARNING_RATE), wrt=is_lora_param)

            # Split model into LoRA and non-LoRA parameters
            self.graphdef, self.lora_params, self.non_lora_params = nnx.split(self.model, is_lora_param, ...)

        logger.info(
            f"Initialized base model {self.config.base_model} with max_lora_adapters={self.config.max_lora_adapters}, max_lora_rank={self.config.max_lora_rank}"
        )

        self._create_loss_and_grad_fn()

    def _create_loss_and_grad_fn(self):
        """Compile and cache the loss function to avoid re-jitting on every call."""

        def loss_for_lora(lora_state, input_ids, attention_mask, adapter_indices, target_ids, loss_mask):
            merged_model = nnx.merge(self.graphdef, lora_state, self.non_lora_params)
            logits = merged_model(input_ids, attention_mask=attention_mask, adapter_indices=adapter_indices)[
                "logits"
            ]  # [B, T, V]
            per_token_losses = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=target_ids, where=loss_mask
            )  # [B, T]
            per_seq_loss = (per_token_losses * loss_mask).sum(axis=-1) / loss_mask.sum(axis=-1)
            # Return sum of losses (we'll divide gradients by per-adapter batch size later)
            return per_seq_loss.sum(), (logits, per_token_losses)

        loss_and_grad_fn = jax.value_and_grad(loss_for_lora, has_aux=True)

        if self.config.enforce_eager:
            # Disable JIT compilation for debugging
            self._loss_and_grad_fn = loss_and_grad_fn
        else:
            # Extract state once to get the pytree structure and compute the partition
            state = nnx.state(self.lora_params)
            state_partition_spec = nnx.get_partition_spec(state)
            # Create NamedSharding objects that tell us how models parameters and inputs should be sharded
            state_shardings = jax.tree.map(lambda spec: jax.NamedSharding(self.mesh, spec), state_partition_spec)
            replicated = jax.NamedSharding(self.mesh, jax.P(None))
            scalar = jax.NamedSharding(self.mesh, jax.P())
            self._loss_and_grad_fn = jax.jit(
                loss_and_grad_fn,
                # One input sharding parameter for each argument of loss_for_lora
                in_shardings=(state_shardings,) + (replicated,) * 5,
                # One output sharding parameter for each return value of loss_for_lora
                out_shardings=((scalar, (replicated, replicated)), state_shardings),
            )

    def _micro_batch_size(self, total: int) -> int:
        """Return effective micro-batch size; 0/absent => disabled (use full fused batch)."""
        mb = self.config.micro_batch_size
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
    ) -> tuple[jax.Array, jax.Array, nnx.State]:
        """Run forward+backward on a batch of inputs."""
        lora_state = nnx.state(self.lora_params)
        seq_len = input_ids.shape[1]
        with jax.set_mesh(self.mesh), self._jit_timing_context(seq_len):
            (_, (logits, per_token_losses)), lora_grads = self._loss_and_grad_fn(
                lora_state, input_ids, attention_mask, adapter_indices, target_ids, loss_mask
            )
        logprobs = jax.nn.log_softmax(logits, axis=-1)  # [B, T, V]
        target_logprobs = jnp.take_along_axis(logprobs, target_ids[..., None], axis=-1).squeeze(-1)  # [B, T]
        return per_token_losses, target_logprobs, lora_grads

    def _accumulate_grads(self, lora_grads: nnx.State, example_model_ids: list[str]) -> None:
        """
        Accumulate adapter-wise gradient sums and example counts.
        """
        for model_id, count in Counter(example_model_ids).items():
            idx = self.models[model_id].adapter_index
            # Extract gradient sum for this adapter
            grad_sum = jax.tree.map(lambda g: g[idx], lora_grads)
            accumulator = self.accumulated_grads[model_id]
            accumulator.add(grad_sum, count)

    def find_batchable_forward_backward(self, session: Session) -> list[FutureDB]:
        """Find all forward_backward ops that come before any optim_step for their model.

        Uses look-ahead scheduling: for each model, only returns forward_backward operations
        that have no optim_step blocking them in the queue.

        Args:
            session: Database session

        Returns:
            List of FutureDB objects that can be safely batched together
        """
        # Find the earliest pending optim_step per model (these act as barriers)
        optim_barriers_query = (
            select(FutureDB.model_id, func.min(FutureDB.request_id).label("barrier_id"))
            .where(FutureDB.request_type == types.RequestType.OPTIM_STEP)
            .where(FutureDB.status == RequestStatus.PENDING)
            .group_by(FutureDB.model_id)
        )
        optim_barriers = session.exec(optim_barriers_query).all()
        barriers = dict(optim_barriers)

        # Get all pending forward_backward operations ordered by request_id
        fwd_bwd_query = (
            select(FutureDB)
            .where(FutureDB.request_type == types.RequestType.FORWARD_BACKWARD)
            .where(FutureDB.status == RequestStatus.PENDING)
            .order_by(FutureDB.request_id)
        )
        fwd_bwd_ops = session.exec(fwd_bwd_query).all()

        # Filter: only include ops that come before their model's optim barrier
        batchable = [op for op in fwd_bwd_ops if op.model_id not in barriers or op.request_id < barriers[op.model_id]]

        return batchable

    def process_create_model(self, model_id: str, request_data: types.CreateModelInput) -> types.CreateModelOutput:
        """Create and initialize a model."""
        # Assign adapter index for this model_id
        adapter_index = max((m.adapter_index for m in self.models.values()), default=-1) + 1

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
        self.accumulated_grads[model_id] = AccumulatedGradients(grad_sum=None, denominator=0)

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
        self, requests: list[tuple[FutureDB, str, types.ForwardBackwardInput]]
    ) -> dict[str, types.ForwardBackwardOutput | types.ForwardBackwardError]:
        """Process multiple forward_backward requests in a single batch.

        Args:
            requests: List of (future, model_id, request_data) tuples

        Returns:
            Dict mapping request_id -> result_data or error info
        """
        if not requests:
            return {}

        results = {}
        valid_requests = []

        # Filter out invalid requests and mark them as failed
        for future, model_id, request_data in requests:
            if model_id not in self.models:
                results[future.request_id] = types.ForwardBackwardError(
                    error=f"Model {model_id} not loaded",
                    status="failed",
                )
            else:
                valid_requests.append((future, model_id, request_data))

        if not valid_requests:
            return results

        # Collect all examples and their metadata
        all_input_ids = []
        all_targets = []
        all_token_weights = []
        all_adapter_indices = []
        example_model_ids = []  # map each example to its model_id
        request_batch_slices = []  # Track which examples belong to which request

        current_batch_idx = 0
        for future, model_id, request_data in valid_requests:
            adapter_index = self.models[model_id].adapter_index
            forward_backward_input = request_data.forward_backward_input
            data = forward_backward_input["data"]

            request_start = current_batch_idx
            for item in data:
                tokens = [t for chunk in item["model_input"]["chunks"] for t in chunk["tokens"]]
                all_input_ids.append(tokens)
                target_tokens = item["loss_fn_inputs"]["target_tokens"]["data"]
                all_targets.append(target_tokens)
                weights = item["loss_fn_inputs"]["weights"]["data"]
                all_token_weights.append(weights)
                all_adapter_indices.append(adapter_index)
                example_model_ids.append(model_id)
                current_batch_idx += 1

            request_batch_slices.append((future.request_id, model_id, request_start, current_batch_idx))

        # Pad sequences to same length. Also bin it so the JIT has to compile fewer kernels.
        max_len = round_up_seq_len(max(len(seq) for seq in all_input_ids))

        input_ids = jnp.array([seq + [0] * (max_len - len(seq)) for seq in all_input_ids], dtype=jnp.int32)
        target_ids = jnp.array([seq + [0] * (max_len - len(seq)) for seq in all_targets], dtype=jnp.int32)
        adapter_indices = jnp.array(all_adapter_indices, dtype=jnp.int32)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = jnp.array(
            [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in all_input_ids], dtype=jnp.int32
        )
        loss_mask = jnp.array(
            [all_token_weights[i] + [0] * (max_len - len(all_input_ids[i])) for i in range(len(all_token_weights))],
            dtype=jnp.int32,
        )

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

        # Apply optimizer update -- going forward we need to figure out how to use different learning rates per adapter
        adam_params = request_data.adam_params
        assert adam_params.lr == LEARNING_RATE, f"Currently we only support a fixed learning rate {LEARNING_RATE}"
        self.optimizer.update(self.lora_params, full_lora_grads)

        # Clear accumulated gradients
        self.accumulated_grads[model_id].reset()

        logger.info(f"Applied optimizer step for model {model_id} (adapter {adapter_index})")
        return types.OptimStepOutput()

    def process_save_weights_for_sampler(
        self, model_id: str, request_data: types.SaveWeightsForSamplerInput
    ) -> types.SaveWeightsForSamplerOutput:
        """Process a save_weights_for_sampler request and save model weights."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        adapter_index = self.models[model_id].adapter_index

        # Make sure the user cannot store checkpoints in places like ../../<important file>
        checkpoint_id = Path(request_data.path).name
        output_dir = self.config.checkpoints_base / model_id / checkpoint_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect LoRA rank for each layer and then the LoRA parameters for adapter_index

        layer_rank = {
            path[:-2]: int(node[adapter_index])
            for path, node in jax.tree.flatten_with_path(self.non_lora_params)[0]
            if len(path) >= 2 and getattr(path[-2], "key", None) == "lora_ranks"
        }

        def extract_adapter_params(path, p):
            rank = layer_rank[path[:-2]]
            if path[-2].key == "lora_A":
                return p[adapter_index, :, :rank]
            elif path[-2].key == "lora_B":
                return p[adapter_index, :rank, :]
            else:
                return p[adapter_index]

        adapter_lora_params = jax.tree.map_with_path(extract_adapter_params, self.lora_params)

        # Save only the LoRA adapter weights
        save_checkpoint(self.model_config, adapter_lora_params, output_dir / "adapter_model.safetensors")

        # Save LoRA config
        lora_config = LoraConfig(
            r=self.models[model_id].lora_config.rank, lora_alpha=self.models[model_id].lora_config.alpha
        )
        lora_config.save_pretrained(output_dir)

        logger.info(f"Saved LoRA adapter weights for model {model_id} (adapter {adapter_index}) to {output_dir}")

        return types.SaveWeightsForSamplerOutput(
            path=f"tinker://{model_id}/{checkpoint_id}",
            type="save_weights_for_sampler",
        )

    def process_single_request(self, request_type: types.RequestType, model_id: str, request_data: dict) -> dict:
        match request_type:
            case types.RequestType.CREATE_MODEL:
                result = self.process_create_model(model_id, types.CreateModelInput.model_validate(request_data))
            case types.RequestType.OPTIM_STEP:
                result = self.process_optim_step(model_id, types.OptimStepInput.model_validate(request_data))
            case types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER:
                result = self.process_save_weights_for_sampler(
                    model_id, types.SaveWeightsForSamplerInput.model_validate(request_data)
                )
            case _:
                raise ValueError(f"Unknown request type: {request_type}")
        return result.model_dump()

    def process_pending_requests(self):
        """Main loop to process pending requests."""
        while True:
            with Session(self.db_engine) as session:
                # Use look-ahead scheduling to find batchable forward_backward operations
                forward_backward_futures = self.find_batchable_forward_backward(session)

                # Get other pending requests (non-forward_backward or those blocked by optim_step)
                statement = (
                    select(FutureDB)
                    .where(FutureDB.status == RequestStatus.PENDING)
                    .where(FutureDB.request_type != types.RequestType.FORWARD_BACKWARD)
                    .order_by(FutureDB.request_id)
                )
                other_futures = session.exec(statement).all()

                # Process forward_backward requests in batch
                if forward_backward_futures:
                    try:
                        batch_requests = [
                            (f, f.model_id, types.ForwardBackwardInput.model_validate(f.request_data))
                            for f in forward_backward_futures
                        ]
                        results = self.process_forward_backward_batch(batch_requests)

                        # Update each future with its result
                        for future in forward_backward_futures:
                            if future.request_id in results:
                                result_data = results[future.request_id]
                                if isinstance(result_data, types.ForwardBackwardError):
                                    future.status = RequestStatus.FAILED
                                else:
                                    future.status = RequestStatus.COMPLETED
                                future.result_data = result_data.model_dump()
                                future.completed_at = datetime.now(timezone.utc)
                                session.add(future)
                                logger.info(f"Completed {future.request_type} request {future.request_id}")

                        session.commit()

                    except Exception as e:
                        logger.exception(f"Error processing forward_backward batch: {e}")
                        # Mark all forward_backward requests in the batch as failed
                        for future in forward_backward_futures:
                            future.result_data = {"error": str(e)}
                            future.status = RequestStatus.FAILED
                            future.completed_at = datetime.now(timezone.utc)
                            session.add(future)
                        session.commit()

                # Process other request types individually (in the future we can also batch independent optim_steps)
                for future in other_futures:
                    try:
                        future.result_data = self.process_single_request(
                            future.request_type, future.model_id, future.request_data
                        )
                        future.status = RequestStatus.COMPLETED
                        future.completed_at = datetime.now(timezone.utc)
                        session.add(future)
                        session.commit()

                        logger.info(f"Completed {future.request_type} request {future.request_id}")

                    except Exception as e:
                        logger.exception(f"Error processing request {future.request_id}: {e}")
                        future.result_data = {"error": str(e)}
                        future.status = RequestStatus.FAILED
                        future.completed_at = datetime.now(timezone.utc)
                        session.add(future)
                        session.commit()

            # Poll every 100ms
            time.sleep(0.1)

    def run(self):
        """Entry point to start the engine."""
        logger.info("Starting background engine...")
        self.process_pending_requests()


def main():
    """Entry point for the background engine."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(filename)s:%(lineno)d] - %(message)s")

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
