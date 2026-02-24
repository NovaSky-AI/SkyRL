#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="run_$(date +%Y%m%d%H)"
ACC_MIN_VALUE=0.58
AVG_NUM_TOKENS_MAX_VALUE=270
LOGPROBS_DIFF_MAX_VALUE=0.035

uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
bash tests/train/gpu_e2e_test/gsm8k_fully_async.sh trainer.run_name=\"$RUN_NAME\"

python ci/get_summary.py --run_name $RUN_NAME --project_name "gsm8k_fully_async_ci" --asserts "eval/all/avg_score >= $ACC_MIN_VALUE" "loss/avg_final_rewards >= $ACC_MIN_VALUE" "generate/avg_num_tokens <= $AVG_NUM_TOKENS_MAX_VALUE" "policy/rollout_train_logprobs_abs_diff_mean <= $LOGPROBS_DIFF_MAX_VALUE"
