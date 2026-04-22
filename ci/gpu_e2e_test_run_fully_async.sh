#!/usr/bin/env bash
set -euo pipefail

# Use only 2500 samples for the CI run
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k --max_train_dataset_length 2500
bash tests/train/gpu_e2e_test/gsm8k_fully_async.sh
