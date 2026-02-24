#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="run_$(date +%Y%m%d%H)"
EVAL_ACC_MIN_VALUE=0.69
TRAIN_ACC_MIN_VALUE=0.69
NUM_TOKENS_MAX_VALUE=232
LOGPROBS_DIFF_MAX_VALUE=0.0104


uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# override run name for bookkeeping
bash tests/train/gpu_e2e_test/gsm8k_colocate.sh  trainer.run_name=\"$RUN_NAME\"

# check if the run is successful
# We check for the following metrics:
# Eval and train accuracy should be greater than the threshold
# Average number of tokens generated should decrease over time
# Policy rollout train logprobs absolute difference should be small
python ci/get_summary.py --run_name $RUN_NAME --project_name "gsm8k_ci" --asserts "eval/all/avg_score >= $EVAL_ACC_MIN_VALUE" "loss/avg_final_rewards >= $TRAIN_ACC_MIN_VALUE" "generate/avg_num_tokens <= $NUM_TOKENS_MAX_VALUE" "policy/rollout_train_logprobs_abs_diff_mean <= $LOGPROBS_DIFF_MAX_VALUE"
