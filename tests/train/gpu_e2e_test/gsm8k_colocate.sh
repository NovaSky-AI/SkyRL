#!/usr/bin/env bash
set -euo pipefail

# Include seconds + PID so the name is unique per invocation. The wandb project is
# shared, so an hour-granular name can collide with a concurrent run on another host,
# and get_summary.py (which picks the newest run by that name) would read the wrong run.
RUN_NAME="run_$(date +%Y%m%d%H%M%S)_$$"
SCRIPT_DIR=$(dirname $(realpath $0))

# Thresholds: approximately 5% allowance from completed DPPO CI runs on Nov 7-9, 2026.
EVAL_ACC_MIN_VALUE=0.69
TRAIN_ACC_MIN_VALUE=0.71
NUM_TOKENS_MAX_VALUE=190
LOGPROBS_DIFF_MAX_VALUE=0.012

# The anyscale job's working_dir is the repo root, so we can use relative paths.
bash examples/train/gsm8k/run_gsm8k.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.project_name=\"gsm8k_ci\" \
  trainer.run_name=\"$RUN_NAME\"

# check if the run is successful
# We check for the following metrics:
# Eval and train accuracy should be greater than the threshold
# Average number of tokens generated should decrease over time
# Policy minibatch versus rollout logprobs absolute difference should be small.
uv run --isolated --extra fsdp $SCRIPT_DIR/get_summary.py --run_name $RUN_NAME --project_name "gsm8k_ci" --wait_for_finished --asserts "eval/all/avg_score >= $EVAL_ACC_MIN_VALUE" "loss/avg_final_rewards >= $TRAIN_ACC_MIN_VALUE" "generate/avg_num_tokens <= $NUM_TOKENS_MAX_VALUE" "policy/minibatch_rollout_logprobs_abs_diff_mean <= $LOGPROBS_DIFF_MAX_VALUE"
