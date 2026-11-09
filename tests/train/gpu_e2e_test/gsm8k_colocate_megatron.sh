#!/usr/bin/env bash
set -euo pipefail

# Unique per invocation (seconds + PID): the shared wandb project means an hour-granular
# name can collide with a concurrent run on another host, and get_summary.py would then
# read the wrong run.
RUN_NAME="run_$(date +%Y%m%d%H%M%S)_$$"
PROJECT_NAME="gsm8k_ci_megatron"
SCRIPT_DIR=$(dirname $(realpath $0))

# TODO (sumanthrh): Thresholds are different for Megatron and FSDP because of differences in batch size/ step size. We should unify the settings
# Thresholds: approximately 5% allowance from completed DPPO CI runs on Nov 6-9, 2026.
EVAL_ACC_MIN_VALUE=0.56
TRAIN_ACC_MIN_VALUE=0.56
NUM_TOKENS_MAX_VALUE=660
LOGPROBS_DIFF_MAX_VALUE=0.0165

# The anyscale job's working_dir is the repo root, so we can use relative paths.
bash examples/train/megatron/run_megatron.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.project_name=\"$PROJECT_NAME\" \
  trainer.run_name=\"$RUN_NAME\"

uv run --isolated --extra fsdp $SCRIPT_DIR/get_summary.py --run_name $RUN_NAME --project_name $PROJECT_NAME --expected_global_step 10 --asserts "eval/all/avg_score >= $EVAL_ACC_MIN_VALUE" "loss/avg_final_rewards >= $TRAIN_ACC_MIN_VALUE" "generate/avg_num_tokens <= $NUM_TOKENS_MAX_VALUE" "policy/minibatch_rollout_logprobs_abs_diff_mean <= $LOGPROBS_DIFF_MAX_VALUE"
