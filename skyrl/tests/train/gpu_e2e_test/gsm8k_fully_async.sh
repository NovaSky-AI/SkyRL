#!/usr/bin/env bash
set -euo pipefail

# The anyscale job's working_dir is the repo root, so we can use relative paths.
bash examples/fully_async/fully_async_run_gsm8k.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.project_name=\"gsm8k_fully_async_ci\" \
  trainer.run_name=\"run_$(date +%Y%m%d%H)\"
