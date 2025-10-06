"""Background engine for processing training requests."""
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from sqlmodel import create_engine, Session, select

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from transformers import AutoConfig
from huggingface_hub import snapshot_download

from tx.tinker.models import FutureDB, ModelDB, DB_PATH, RequestType, RequestStatus
from tx.utils.models import get_dtype, get_model_class, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

# Base path for saving checkpoints
CHECKPOINTS_BASE_PATH = Path("/tmp/tx_checkpoints")


def loss_fn(model, batch):
    """Compute loss for a batch."""
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["target_ids"]
    )
    return loss.mean(), logits


class TinkerEngine:
    """Background engine for processing training requests."""

    def __init__(self, db_path=DB_PATH):
        """Initialize the engine with a database connection."""
        self.db_engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.models = {}  # Store loaded models: model_id -> {"model": model, "optimizer": optimizer, "config": config}
        self.accumulated_grads = {}  # Store accumulated gradients: model_id -> grads

    def create_model(self, model_id: str, base_model: str, lora_config: dict | None = None):
        """Create and initialize a model."""
        config = AutoConfig.from_pretrained(base_model)
        model_class = get_model_class(config)

        # Download model weights from HuggingFace
        checkpoint_path = snapshot_download(base_model, allow_patterns=["*.safetensors"])

        # Create model and load weights
        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = model_class(config, dtype=get_dtype(config.dtype), rngs=nnx.Rngs(0))
            # Initialize optimizer with default Adam settings
            # (TODO: This might not actually be super great, it is worth thinking about how to do that better)
            optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=nnx.Param)
        load_checkpoint(checkpoint_path, config, model)

        self.models[model_id] = {
            "model": model,
            "optimizer": optimizer,
            "config": config
        }
        self.accumulated_grads[model_id] = None
        logger.info(f"Created model {model_id} from {base_model}")

    def process_forward_backward(self, request_id: str, model_id: str, request_data: dict) -> dict:
        """Process a forward_backward request and return real loss and gradients."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        model_info = self.models[model_id]
        model = model_info["model"]

        forward_backward_input = request_data.get("forward_backward_input", {})

        # Extract tokens from examples
        data = forward_backward_input["data"]
        input_ids_list = []
        target_list = []
        for item in data:
            tokens = [t for chunk in item["model_input"]["chunks"] for t in chunk["tokens"]]
            input_ids_list.append(tokens)
            target_tokens = item["loss_fn_inputs"]["target_tokens"]["data"]
            target_list.append(target_tokens)

        # Pad sequences to same length
        max_len = max(len(seq) for seq in input_ids_list)
        padded_inputs = [seq + [0] * (max_len - len(seq)) for seq in input_ids_list]
        padded_targets = [seq + [0] * (max_len - len(seq)) for seq in target_list]

        input_ids = jnp.array(padded_inputs, dtype=jnp.int32)
        target_ids = jnp.array(padded_targets, dtype=jnp.int32)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = jnp.array(
            [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in input_ids_list],
            dtype=jnp.int32
        )

        # Compute gradients
        model.train()
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, {"input_ids": input_ids, "attention_mask": attention_mask, "target_ids": target_ids})

        # Accumulate gradients
        if self.accumulated_grads[model_id] is None:
            self.accumulated_grads[model_id] = grads
        else:
            raise NotImplementedError("Gradient accumulation not yet implemented")

        # Return loss in the expected format
        loss_value = float(loss)
        result = {
            "loss_fn_output_type": "scalar",
            "loss_fn_outputs": [{
                "loss": {
                    "data": [loss_value],
                    "dtype": "float32",
                    "shape": [1]
                }
            }],
            "metrics": {}
        }
        return result

    def process_optim_step(self, request_id: str, model_id: str, request_data: dict) -> dict:
        """Process an optim_step request and apply accumulated gradients."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        model_info = self.models[model_id]
        model = model_info["model"]
        optimizer = model_info["optimizer"]

        # Get accumulated gradients
        grads = self.accumulated_grads.get(model_id)
        if grads is None:
            logger.warning(f"No accumulated gradients for model {model_id}, skipping optimizer step")
            return {}

        adam_params = request_data.get("adam_params", {})
        # TODO: Add weight decay and other parameter handling here
        optimizer.update(model, grads, learning_rate=adam_params["lr"])

        # Clear accumulated gradients
        self.accumulated_grads[model_id] = None

        logger.info(f"Applied optimizer step for model {model_id}")
        return {}

    def process_save_weights_for_sampler(self, request_id: str, model_id: str, request_data: dict) -> dict:
        """Process a save_weights_for_sampler request and save model weights."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        model_info = self.models[model_id]
        model = model_info["model"]
        config = model_info["config"]

        checkpoint_id = request_data.get("path")
        if not checkpoint_id:
            checkpoint_id = f"checkpoint_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        else:
            # Make sure the user cannot store checkpoints in places like ../../<important file>
            checkpoint_id = Path(checkpoint_id).name

        output_dir = CHECKPOINTS_BASE_PATH / model_id / checkpoint_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the model weights and config to disk
        save_checkpoint(config, model, output_dir / "model.safetensors")
        config.save_pretrained(output_dir)
        logger.info(f"Saved weights for sampler for model {model_id} to {output_dir}")

        return {
            "path": f"tinker://{model_id}/{checkpoint_id}",
            "type": "save_weights_for_sampler"
        }

    def process_pending_requests(self):
        """Main loop to process pending requests."""
        while True:
            with Session(self.db_engine) as session:
                # Get all pending requests
                statement = select(FutureDB).where(FutureDB.status == RequestStatus.PENDING)
                pending = session.exec(statement).all()

                for future in pending:
                    try:
                        # Process based on request type
                        if future.request_type == RequestType.CREATE_MODEL:
                            # Get model from database and create it
                            model_statement = select(ModelDB).where(ModelDB.model_id == future.model_id)
                            model_db = session.exec(model_statement).first()
                            if not model_db:
                                raise ValueError(f"Model {future.model_id} not found in database")
                            self.create_model(future.model_id, model_db.base_model, model_db.lora_config)
                            result_data = {
                                "model_id": future.model_id,
                                "base_model": model_db.base_model,
                                "lora_config": model_db.lora_config,
                                "status": "created",
                                "request_id": future.request_id
                            }
                        elif future.request_type == RequestType.FORWARD_BACKWARD:
                            result_data = self.process_forward_backward(
                                future.request_id,
                                future.model_id,
                                future.request_data
                            )
                        elif future.request_type == RequestType.OPTIM_STEP:
                            result_data = self.process_optim_step(
                                future.request_id,
                                future.model_id,
                                future.request_data
                            )
                        elif future.request_type == RequestType.SAVE_WEIGHTS_FOR_SAMPLER:
                            result_data = self.process_save_weights_for_sampler(
                                future.request_id,
                                future.model_id,
                                future.request_data
                            )
                        else:
                            logger.warning(f"Unknown request type: {future.request_type}")
                            continue

                        # Update the future with results
                        future.result_data = result_data
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
    TinkerEngine().run()


if __name__ == "__main__":
    main()
