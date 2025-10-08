import jax.numpy as jnp
from datasets import Dataset
from transformers import PreTrainedTokenizer

from tx.loaders.common import LoaderIterator


def chat(tokenizer: PreTrainedTokenizer, dataset: Dataset, batch_size: int) -> LoaderIterator:
    "Data loader that applies the chat template. It returns an iterator over (batch, metrics) elements."

    for data in dataset.shuffle().iter(batch_size=batch_size):
        batch = tokenizer.apply_chat_template(data["messages"], tokenize=False)
        # We pad to multiples of 512 here so jax needs to compile less different shapes
        batch = tokenizer(batch, return_tensors="np", padding=True, pad_to_multiple_of=512)
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        yield {
            "text": batch["input_ids"][:, :-1],
            "attention_mask": batch["attention_mask"][:, :-1],
            "target": batch["input_ids"][:, 1:],
        }, {"shape": batch["input_ids"].shape, "tokens": batch["attention_mask"].sum()}
