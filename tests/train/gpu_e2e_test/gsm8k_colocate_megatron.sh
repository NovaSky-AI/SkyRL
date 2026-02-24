#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="run_$(date +%Y%m%d%H)"
SCRIPT_DIR=$(dirname $(realpath $0))

# TODO (sumanthrh): Thresholds are different for Megatron and FSDP because of differences in batch size/ step size. We should unify the settings
# Thresholds: 5% allowance from min/max of last 10 CI runs as of 23rd Feb 2026
EVAL_ACC_MIN_VALUE=0.54
TRAIN_ACC_MIN_VALUE=0.52
NUM_TOKENS_MAX_VALUE=665
LOGPROBS_DIFF_MAX_VALUE=0.01764

# The anyscale job's working_dir is the repo root, so we can use relative paths.
bash examples/train/megatron/run_megatron.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.project_name=\"gsm8k_ci_megatron\" \
  trainer.run_name=\"$RUN_NAME\"

python $SCRIPT_DIR/get_summary.py --run_name $RUN_NAME --project_name "gsm8k_ci_megatron" --asserts "eval/all/avg_score >= $EVAL_ACC_MIN_VALUE" "loss/avg_final_rewards >= $TRAIN_ACC_MIN_VALUE" "generate/avg_num_tokens <= $NUM_TOKENS_MAX_VALUE" "policy/rollout_train_logprobs_abs_diff_mean <= $LOGPROBS_DIFF_MAX_VALUE"