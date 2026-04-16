#!/usr/bin/env bash
set -euo pipefail

export _SKYRL_USE_NEW_INFERENCE=1
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
bash tests/train/gpu_e2e_test/gsm8k_colocate.sh
