#!/usr/bin/env bash
set -euo pipefail

# The anyscale job's working_dir is the repo root, so we can use relative paths.
bash examples/megatron/run_megatron.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.project_name=\"gsm8k_ci_megatron\" \
  trainer.run_name=\"run_$(date +%Y%m%d%H)\"
