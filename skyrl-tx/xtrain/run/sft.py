import os

from datasets import IterableDataset, load_dataset
from flax import nnx
import optax
import safetensors.numpy
import typer

from xtrain.models import Mnist

app = typer.Typer()


def save_checkpoint(state: nnx.State, filename: str | os.PathLike) -> None:
    params = nnx.to_flat_state(state)
    tensor_dict = {".".join(k): v for k, v in params if "rngs" not in k}
    safetensors.numpy.save_file(tensor_dict, filename)


def loss_fn(model, batch):
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    )
    return loss.mean(), logits


@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(model, grads)

    
@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", help="HuggingFace dataset to use for training.")
) -> None:
    train_dataset = load_dataset(dataset, split="train")
    
    model = Mnist(rngs=nnx.Rngs(0))

    optimizer = nnx.Optimizer(
        model, optax.adamw(0.005, 0.9), wrt=nnx.Param
    )
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    ds: IterableDataset = train_dataset.with_format("numpy") # ty: ignore
    for step, batch in enumerate(ds.iter(batch_size=32)):
        model.train()
        input_batch = {
            "image": 1.0 * batch["image"][:,:,:,None] / 255.0,
            "label": batch["label"]
        }
        train_step(model, optimizer, metrics, input_batch)

    for metric, value in metrics.compute().items():
        print(metric, value)

    save_checkpoint(nnx.state(model), "checkpoint.safetensors")


if __name__ == "__main__":
    app()

