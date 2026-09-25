# Geometry-3K Multi-Turn VLM RL

Multi-turn GRPO on the [Geometry-3K dataset](https://huggingface.co/datasets/hiyouga/geometry3k) with `Qwen/Qwen3-VL-8B-Instruct`. The model sees a geometry diagram plus a text question, reasons across up to 3 turns, and checks candidate answers with a `calc_score` tool before committing to a final `\boxed{}` answer. Reward is binary (1.0 correct / 0.0 otherwise).

See [the docs](https://docs.skyrl.ai/docs/examples/geometry3k) and the [Vision-Language RL tutorial](https://docs.skyrl.ai/docs/tutorials/vision_language_rl) for a walkthrough.

## Quick start

```bash
bash examples/train/geometry3k/run_geometry3k.sh
```

The dataset auto-generates on first run. Override via env vars:

```bash
LOGGER=wandb bash examples/train/geometry3k/run_geometry3k.sh
DATA_DIR=/path/to/data bash examples/train/geometry3k/run_geometry3k.sh
EXPORT_PATH=/path/to/exports CKPT_PATH=/path/to/ckpts bash examples/train/geometry3k/run_geometry3k.sh
```

On a multi-node Ray cluster, put `DATA_DIR`, `EXPORT_PATH` and `CKPT_PATH` on storage shared by all nodes.

## Variants

- `run_geometry3k_lora.sh`: LoRA on the FSDP backend.
- `run_geometry3k_megatron.sh`: Megatron backend (defaults to `MEGATRON_TP=2`, `MEGATRON_PP=1`).
