#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="run_$(date +%Y%m%d%H)"
# TODO (sumanthrh): Thresholds are different for Megatron and FSDP because of differences in batch size/ step size. We should unify the settings
EVAL_ACC_MIN_VALUE=0.54
TRAIN_ACC_MIN_VALUE=0.52
NUM_TOKENS_MAX_VALUE=665
LOGPROBS_DIFF_MAX_VALUE=0.01764

uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k --max_train_dataset_length 1280
bash tests/train/gpu_e2e_test/gsm8k_colocate_megatron.sh trainer.run_name=\"$RUN_NAME\"

python ci/get_summary.py --run_name $RUN_NAME --project_name "gsm8k_ci_megatron" --asserts "eval/all/avg_score >= $EVAL_ACC_MIN_VALUE" "loss/avg_final_rewards >= $TRAIN_ACC_MIN_VALUE" "generate/avg_num_tokens <= $NUM_TOKENS_MAX_VALUE" "policy/rollout_train_logprobs_abs_diff_mean <= $LOGPROBS_DIFF_MAX_VALUE"
