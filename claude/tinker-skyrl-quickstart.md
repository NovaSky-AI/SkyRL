# Running Tinker Cookbook on SkyRL

A quick guide to running [tinker-cookbook](https://github.com/thinkingmachines/tinker-cookbook) recipes using SkyRL as the backend.

## Prerequisites

- Linux machine with NVIDIA GPUs (tested on 4xL4)
- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

### 1. Clone the repositories

```bash
# Clone SkyRL
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL
git checkout tyler/tinker-sampling-main  # or main once PR #999 is merged

# Clone tinker-cookbook (in a separate directory)
cd ~
git clone https://github.com/thinkingmachines/tinker-cookbook.git
```

### 2. Start the Tinker API Server

```bash
cd ~/SkyRL/skyrl-tx

# Clean any previous state
rm -f tx/tinker/tinker.db

# Start the server
uv run --extra skyrl_train --extra tinker -m tx.tinker.api \
    --base-model "Qwen/Qwen3-0.6B" \
    --backend skyrl_train
```

The server takes ~2 minutes to initialize. Wait until you see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Run a Tinker Cookbook Recipe

In a new terminal:

```bash
cd ~/tinker-cookbook

TINKER_API_KEY=tml-test uv run --with tinker --with datasets --with torch \
    python -m tinker_cookbook.recipes.rl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B" \
    batch_size=8 \
    group_size=4 \
    lora_rank=32 \
    max_tokens=128 \
    save_every=5 \
    log_path="/tmp/tinker-rl-test"
```

## Supported Models

Use models from the [Qwen3 family](https://huggingface.co/Qwen):
- `Qwen/Qwen3-0.6B` - Small, fast for testing
- `Qwen/Qwen3-1.7B` - Medium
- `Qwen/Qwen3-4B` - Larger, needs more VRAM
- `Qwen/Qwen3-8B` - Requires 4+ GPUs

## Configuration Options

### Server options

| Option | Description | Default |
|--------|-------------|---------|
| `--base-model` | HuggingFace model ID | Required |
| `--backend` | Backend type (`skyrl_train` or `jax`) | `jax` |
| `--checkpoints-base` | Checkpoint storage path | `/tmp/tx_checkpoints` |

### rl_loop.py options

| Option | Description | Default |
|--------|-------------|---------|
| `base_url` | Tinker API URL | Required |
| `model_name` | Must match server's base-model | Required |
| `batch_size` | Questions per batch | 8 |
| `group_size` | Rollouts per question | 4 |
| `lora_rank` | LoRA adapter rank | 32 |
| `max_tokens` | Max generation length | 128 |
| `save_every` | Checkpoint frequency | 5 |
| `log_path` | Output directory | Required |

## Troubleshooting

### "Model already exists" error
```bash
rm ~/SkyRL/skyrl-tx/tx/tinker/tinker.db
# Restart the server
```

### Out of memory
Reduce `batch_size` and `group_size`:
```bash
batch_size=4 group_size=2
```

### Server won't start
Check GPU availability:
```bash
nvidia-smi
```

### Disk space errors
Clean up checkpoints:
```bash
rm -rf /tmp/tx_checkpoints/*
rm -rf /tmp/tinker-rl-test/*
```

## Output

After running, you'll find:
- `metrics.jsonl` - Training metrics per batch
- `checkpoints.jsonl` - Saved checkpoint paths
- `logs.log` - Detailed logs

Example metrics:
```json
{"step": 0, "progress/batch": 0, "optim/lr": 4e-05, "reward/total": 0.0, "time/total": 19.47}
```

## Other Recipes

The same setup works for other tinker-cookbook recipes:
- `tinker_cookbook.recipes.sft` - Supervised fine-tuning
- `tinker_cookbook.recipes.dpo` - Direct preference optimization

Check the [tinker-cookbook docs](https://github.com/thinkingmachines/tinker-cookbook) for more.

## Links

- [SkyRL Repository](https://github.com/NovaSky-AI/SkyRL)
- [Tinker Cookbook](https://github.com/thinkingmachines/tinker-cookbook)
- [Tinker API Docs](https://tinker-docs.thinkingmachines.ai)
