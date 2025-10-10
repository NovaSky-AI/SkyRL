"""Background engine for processing training requests."""

import time
import logging
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
from tx.utils.models import get_dtype, get_model_class, save_checkpoint, load_checkpoint
from tx.layers.lora import update_adapter_config
from peft import LoraConfig

logger = logging.getLogger(__name__)

LEARNING_RATE = 1e-4


def round_up_to_multiple(num: int, multiple: int) -> int:
    return ((num + multiple - 1) // multiple) * multiple


class TinkerEngine:
    """Background engine for processing training requests."""

    def __init__(
        self,
        base_model_name: str,
        checkpoints_base_path: str,
        max_lora_adapters: int,
        max_lora_rank: int,
        db_path=DB_PATH,
    ):
        """Initialize the engine with a database connection and base model."""
        self.db_engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.base_model_name = base_model_name  # Single base model for this engine
        self.checkpoints_base_path = checkpoints_base_path  # Location where checkpoints will be stored
        self.models: dict[str, types.ModelMetadata] = {}  # Store LoRA model metadata
        self.accumulated_grads = {}  # Store accumulated gradients per LoRA adapter: model_id -> grads
        self.max_lora_adapters = max_lora_adapters  # Maximum number of LoRA adapters
        self.max_lora_rank = max_lora_rank  # Maximum LoRA rank

        # Initialize the shared base model
        self.config = AutoConfig.from_pretrained(self.base_model_name)

        # Configure LoRA settings
        self.config.max_lora_adapters = self.max_lora_adapters
        self.config.max_lora_rank = self.max_lora_rank

        model_class = get_model_class(self.config)

        # Download model weights from HuggingFace
        checkpoint_path = snapshot_download(self.base_model_name, allow_patterns=["*.safetensors"])

        # Create model and load weights
        mesh = jax.make_mesh((1, 2), ("dp", "tp"))
        with jax.set_mesh(mesh):
            self.model = model_class(self.config, dtype=get_dtype(self.config.dtype), rngs=nnx.Rngs(0))
            load_checkpoint(checkpoint_path, self.config, self.model)

            # Create optimizer that only targets LoRA A and B parameters
            def is_lora_param(path, value):
                return any(name in path for name in ["lora_A", "lora_B"])

            self.optimizer = nnx.Optimizer(self.model, optax.adamw(LEARNING_RATE), wrt=is_lora_param)

            # Split model into LoRA and non-LoRA parameters
            self.graphdef, self.lora_params, self.non_lora_params = nnx.split(self.model, is_lora_param, ...)

        logger.info(
            f"Initialized base model {self.base_model_name} with max_lora_adapters={max_lora_adapters}, max_lora_rank={max_lora_rank}"
        )

        self._compile_loss_function()

    def _compile_loss_function(self):
        """Compile and cache the loss function to avoid re-jitting on every call.

        The function captures graphdef and non_lora_params in a closure, which is fine
        because these don't change. We compile it once and reuse it.
        """
        graphdef = self.graphdef
        non_lora_params = self.non_lora_params

        def _loss_fn(lora_params, input_ids, attention_mask, adapter_indices, target_ids, loss_mask):
            merged_model = nnx.merge(graphdef, lora_params, non_lora_params)
            logits = merged_model(input_ids, attention_mask=attention_mask, adapter_indices=adapter_indices)["logits"]
            # Compute per-example losses (don't average yet)
            per_token_losses = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=target_ids, where=loss_mask
            )
            # Average over sequence length for each example
            per_example_losses = per_token_losses.mean(axis=-1)
            # Return mean loss for gradient computation, but also return per-token losses
            return per_example_losses.mean(), (logits, per_token_losses)

        # Compile once and store
        self._compiled_loss_fn = nnx.jit(_loss_fn)

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

        if adapter_index >= self.max_lora_adapters:
            raise ValueError(f"Maximum number of LoRA adapters ({self.max_lora_adapters}) reached")

        # Extract LoRA rank and alpha from config
        lora_rank = request_data.lora_config.rank
        lora_alpha = request_data.lora_config.alpha

        # Validate rank doesn't exceed max
        if not (0 < lora_rank <= self.max_lora_rank):
            raise ValueError(f"LoRA rank {lora_rank} must be between 1 and {self.max_lora_rank}")

        self.models[model_id] = types.ModelMetadata(
            adapter_index=adapter_index,
            lora_config=request_data.lora_config,
        )
        self.accumulated_grads[model_id] = None

        # Update the adapter's rank and scaling in all LoRA layers
        update_adapter_config(self.model, adapter_index, lora_rank, lora_alpha)

        logger.info(
            f"Created LoRA model {model_id} with adapter index {adapter_index}, rank {lora_rank}, alpha {lora_alpha}"
        )

        return types.CreateModelOutput(
            model_id=model_id,
            base_model=self.base_model_name,
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

        # Collect all examples and their adapter indices
        all_input_ids = []
        all_targets = []
        all_token_weights = []
        all_adapter_indices = []
        request_batch_slices = []  # Track which batch elements belong to which request

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
                current_batch_idx += 1

            request_batch_slices.append((future.request_id, model_id, request_start, current_batch_idx))

        # Pad sequences to same length and bin it so the JIT has to compile fewer kernels
        max_len = round_up_to_multiple(max(len(seq) for seq in all_input_ids), 512)

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

        # Compute per-example losses and gradients using the cached compiled function
        # This avoids re-jitting on every call
        loss_and_grad_fn = nnx.value_and_grad(self._compiled_loss_fn, has_aux=True)
        (avg_loss, (logits, per_token_losses)), lora_grads = loss_and_grad_fn(
            self.lora_params, input_ids, attention_mask, adapter_indices, target_ids, loss_mask
        )

        # Compute logprobs for the target tokens
        all_logprobs = jax.nn.log_softmax(logits, axis=-1)  # [B, T, V]
        target_logprobs = jnp.take_along_axis(all_logprobs, target_ids[..., None], axis=-1)  # [B, T, 1]
        target_logprobs = target_logprobs.squeeze(-1)  # [B, T]

        # Extract and accumulate gradients for each model_id's specific adapter
        for request_id, model_id, start_idx, end_idx in request_batch_slices:
            adapter_index = self.models[model_id].adapter_index

            # Extract gradients for this specific adapter index
            adapter_grads = jax.tree.map(lambda g: g[adapter_index], lora_grads)

            if self.accumulated_grads[model_id] is None:
                self.accumulated_grads[model_id] = adapter_grads
            else:
                raise NotImplementedError("Gradient accumulation not yet implemented")

        # Compute per-request results with correct per-request losses
        for request_id, model_id, start_idx, end_idx in request_batch_slices:
            loss_fn_outputs = []
            # Compute per-example losses
            for i in range(start_idx, end_idx):
                # Trim padding, and extract losses for this example's tokens
                seq_len = len(all_input_ids[i])
                token_losses = per_token_losses[i, :seq_len].astype(jnp.float32)
                token_logprobs = target_logprobs[i, :seq_len].astype(jnp.float32)
                loss_fn_outputs.append(
                    {
                        "elementwise_loss": {"data": token_losses.tolist(), "dtype": "float32", "shape": [seq_len]},
                        "logprobs": {"data": token_logprobs.tolist(), "dtype": "float32", "shape": [seq_len]},
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
        adapter_grads = self.accumulated_grads.get(model_id)
        if adapter_grads is None:
            logger.warning(f"No accumulated gradients for model {model_id}, skipping optimizer step")
            return types.OptimStepOutput()

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
        self.accumulated_grads[model_id] = None

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
        output_dir = Path(self.checkpoints_base_path) / model_id / checkpoint_id
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
        save_checkpoint(self.config, adapter_lora_params, output_dir / "adapter_model.safetensors")

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
    from optparse import OptionParser

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(filename)s:%(lineno)d] - %(message)s")

    parser = OptionParser()
    parser.add_option(
        "--base-model", dest="base_model", help="Base model name (e.g., Qwen/Qwen3-0.6B)", metavar="MODEL"
    )
    parser.add_option(
        "--checkpoints-base-path",
        dest="checkpoints_base_path",
        help="Base path where checkpoints will be stored",
        metavar="PATH",
    )
    parser.add_option(
        "--max-lora-adapters",
        dest="max_lora_adapters",
        type="int",
        default=1,
        help="Maximum number of LoRA adapters (default: 32)",
        metavar="NUM",
    )
    parser.add_option(
        "--max-lora-rank",
        dest="max_lora_rank",
        type="int",
        default=4,
        help="Maximum LoRA rank (default: 32)",
        metavar="RANK",
    )

    (options, args) = parser.parse_args()

    if not options.base_model:
        parser.error("--base-model is required")
    if not options.checkpoints_base_path:
        parser.error("--checkpoints-base-path is required")

    TinkerEngine(
        base_model_name=options.base_model,
        checkpoints_base_path=options.checkpoints_base_path,
        max_lora_adapters=options.max_lora_adapters,
        max_lora_rank=options.max_lora_rank,
    ).run()


if __name__ == "__main__":
    main()
