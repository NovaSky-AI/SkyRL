# Running on GPUs

We assume your are logged into a single GPU node with one or more GPUs.
Multi-node instructions will be added later.

## Setting up tx

Install `uv` and clone the `tx` repository with

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/tx-project/tx
cd tx
```

## Starting the training

Next, download the dataset with

```bash
uv run --with huggingface_hub hf download Qwen/Qwen3-4B --local-dir /tmp/qwen3
```

You can then start the training with

```bash
uv run --extra gpu --with jinja2 tx train --model Qwen/Qwen3-4B --dataset HuggingFaceH4/ultrachat_200k --loader tx.loaders.chat --split train_sft --output-dir /tmp/ultrachat --batch-size 8 --load-checkpoint-path /tmp/qwen3 --tp-size 8
```

In this example we assume you have 8 GPUs in the node. If you have a different number of GPUs, you can modify the `--tp-size` parameter appropriately.

See the full set of options of `tx` in [the CLI reference](../reference.md).
