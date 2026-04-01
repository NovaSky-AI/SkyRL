#!/usr/bin/env bash
# Lightweight setup for GSM8K GRPO experiments.
# Does NOT require AWS credentials, Fleet API key, or OpenEnv.
set -eux

# --- Fix Ray binary permissions (some cloud images strip +x) ---
for f in .venv/bin/ray .venv/lib/python*/site-packages/ray/core/src/ray/raylet/raylet; do
  [ -f "$f" ] && chmod +x "$f" 2>/dev/null || true
done

# --- Python environment ---
if [ -d ".venv" ]; then
  echo "Virtual environment already exists, reusing"
else
  uv venv --python 3.12 --seed
fi
source .venv/bin/activate
uv sync --extra fsdp
uv pip install wandb

# --- Download and prepare GSM8K dataset ---
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
echo "Preparing GSM8K dataset at $DATA_DIR..."
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir "$DATA_DIR"
echo "GSM8K dataset ready."
