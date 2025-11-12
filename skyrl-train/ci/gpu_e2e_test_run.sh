#!/usr/bin/env bash
set -euo pipefail

uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
bash tests/gpu/gpu_e2e_test/gsm8k_colocate.sh
