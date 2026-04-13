# Geometry-3K Multi-Modal RL Example

This example demonstrates multi-modal reinforcement learning on the Geometry-3K dataset using SkyRL.

## Dataset

The [Geometry-3K dataset](https://huggingface.co/datasets/hiyouga/geometry3k) contains 3,002 geometry problems with diagrams. Each problem includes:
- **images**: List of geometry diagrams (PIL images)
- **problem**: Problem text that may reference the image(s)
- **answer**: Ground truth answer

## Setup

Run training (the dataset is auto-generated on first run if not already present):

```bash
NUM_GPUS=4 bash examples/train/geometry3k/run_geometry3k.sh
```

To generate the dataset separately:

```bash
uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir ~/data/geometry_3k
```

## Configuration Options

You can override defaults via environment variables:

```bash
# Custom data directory
DATA_DIR=/path/to/data bash examples/train/geometry3k/run_geometry3k.sh

# Enable W&B logging
LOGGER=wandb bash examples/train/geometry3k/run_geometry3k.sh

```

Or pass additional Hydra overrides:

```bash
bash examples/train/geometry3k/run_geometry3k.sh trainer.epochs=50 generator.n_samples_per_prompt=8
```

## Environment

The `geometry3k` environment evaluates model responses against ground truth answers:

- **Reward**: 1.0 for correct answer, 0.0 otherwise
- **Answer extraction**: Extracts answer from `\boxed{}` or `<tool_call>` tags
- **Normalization**: Case-insensitive comparison with punctuation handling
- **Numeric support**: Handles numerical answers with tolerance

## Model

By default, uses `Qwen/Qwen3-VL-8B-Instruct` which is a vision-language model capable of processing images.

## Prompt Format

The prompt template asks the model to:
1. Think through the problem in `<think>...</think>` tags
2. Check answers via `<tool_call>{"name": "calc_score", ...}</tool_call>`
3. Provide the final answer as `\boxed{$Answer}`
