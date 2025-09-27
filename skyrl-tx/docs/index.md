# tx: Transformers x-platform

tx is a simple but powerful cross-platform training library.

## Quickstart

Here is an example of how to fine-tune a model on a HuggingFace dataset:

```bash
# Download Qwen3-8B checkpoint
uv run --with huggingface_hub hf download Qwen/Qwen3-4B --local-dir /tmp/qwen3

# Fine-tune the model on a chat dataset
uv run --with jinja2 tx train --model Qwen/Qwen3-4B --dataset HuggingFaceH4/ultrachat_200k --loader tx.loaders.chat --split train_sft --output-dir /tmp/ultrachat --batch-size 8 --load-checkpoint-path /tmp/qwen3
```
