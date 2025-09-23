from functools import partial
from pathlib import Path
import sys

from datasets import load_dataset
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from transformers import AutoConfig, AutoTokenizer
import typer

from tx.utils.models import FrozenModelConfig, get_dtype, get_model_class, save_checkpoint
from tx.utils.log import add_file_handler, logger

app = typer.Typer()


@partial(nnx.jit, static_argnames=["config", "model_class"])
def create_model(config: FrozenModelConfig, model_class) -> nnx.Module:
    config = config.unfreeze()
    model = model_class(config, dtype=get_dtype(config.dtype), rngs=nnx.Rngs(0))
    state = nnx.state(model)
    sharded_state = jax.lax.with_sharding_constraint(state, nnx.get_partition_spec(state))
    nnx.update(model, sharded_state)
    return model


def loss_fn(model, batch):
    logits = model(batch["text"], attention_mask=batch["attention_mask"])["logits"]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["target"]
    )
    return loss.mean(), logits


@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    gradnorm = optax.global_norm(grads)
    optimizer.update(model, grads)
    return loss, gradnorm


def train(
    model_name: str = typer.Option(..., "--model", help="HuggingFace model ID or local model path"),
    dataset: str = typer.Option(..., "--dataset", help="HuggingFace dataset to use for training"),
    output_dir: Path = typer.Option(..., "--output-dir", help="The output directory where the model predictions and checkpoints will be written"),
    save_steps: int = typer.Option(500, "--save-steps", help="Number of steps between checkpoints"),
    max_steps: int | None = typer.Option(None, "--max-steps", help="The maximum number of training steps"),
    per_device_batch_size: int = typer.Option(..., "--per-device-batch-size", help="Batch size per device accelerator for training"),
    tp_size: int = typer.Option(1, "--tp-size", help="Tensor parallelism degree to use for the model"),
) -> None:
    if not jax._src.xla_bridge.backends_are_initialized():
        jax.config.update('jax_num_cpu_devices', tp_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_dir / "tx.log")
    logger.info(f"tx was invoked with 'tx {' '.join(sys.argv[1:])}'")

    train_dataset = load_dataset(dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model_class = get_model_class(config)

    mesh = jax.make_mesh((1, tp_size), ("dp", "tp"))
    with jax.set_mesh(mesh):
        model = create_model(FrozenModelConfig(config), model_class)

    optimizer = nnx.Optimizer(
        model, optax.adamw(0.002, weight_decay=0.1), wrt=nnx.Param
    )

    num_steps = len(train_dataset) / per_device_batch_size
    for step, data in enumerate(train_dataset.iter(batch_size=per_device_batch_size)):
        if max_steps and step >= max_steps:
            break

        # We pad to multiples of 128 here so jax needs to compile less different shapes
        batch = tokenizer(data["text"], return_tensors="np", padding=True, pad_to_multiple_of=128)
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        model.train()
        input_batch = {
            "text": batch["input_ids"][:,:-1],
            "attention_mask": batch["attention_mask"][:,:-1],
            "target": batch["input_ids"][:, 1:],
        }
        loss, gradnorm = train_step(model, optimizer, input_batch)
        logger.info(f"step: {step}, epoch: {step / num_steps :.2e}, shape: {batch['input_ids'].shape}, tokens: {batch['attention_mask'].sum()}, gradnorm: {gradnorm.item() :5.2f}, loss: {loss.item() :5.2f}")

        if step % save_steps == 0:
            logger.info(f"Saving checkpoint to {output_dir}")
            save_checkpoint(config, model, output_dir / "model.safetensors")

    logger.info(f"Saving final checkpoint to {output_dir}")
    save_checkpoint(config, model, output_dir / "model.safetensors")
