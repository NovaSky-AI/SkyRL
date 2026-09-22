#!/usr/bin/env bash
set -euo pipefail

# Unique per invocation (seconds + PID): the shared wandb project means an hour-granular
# name can collide with a concurrent run on another host, and get_summary.py would then
# read the wrong run.
RUN_NAME="run_$(date +%Y%m%d%H%M%S)_$$"

SCRIPT_DIR=$(dirname $(realpath $0))
# Thresholds: 5% allowance from min/max of every CI run since 20th Jul 2026 (69 runs as of
# 18th Aug 2026). The window starts at #1929 (eval metrics logged at the right step), which
# is the last change to shift these numbers. Rebaselined after the async codepath changes:
# stale KV cache reuse on weight sync (#1798), policy_loss_type=rollout_is (#1850), and the
# per-step cache salt (#1836).
EVAL_ACC_MIN_VALUE=0.50
TRAIN_ACC_MIN_VALUE=0.198
AVG_NUM_TOKENS_MAX_VALUE=285
LOGPROBS_DIFF_MAX_VALUE=0.0193

# The anyscale job's working_dir is the repo root, so we can use relative paths.
bash examples/train/fully_async/fully_async_run_gsm8k.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.run_name=\"$RUN_NAME\" trainer.project_name=\"gsm8k_fully_async_ci\"

uv run --isolated --extra fsdp $SCRIPT_DIR/get_summary.py --run_name $RUN_NAME --project_name "gsm8k_fully_async_ci" --asserts "eval/all/avg_score >= $EVAL_ACC_MIN_VALUE" "loss/avg_final_rewards >= $TRAIN_ACC_MIN_VALUE" "generate/avg_num_tokens <= $AVG_NUM_TOKENS_MAX_VALUE" "policy/rollout_train_logprobs_abs_diff_mean <= $LOGPROBS_DIFF_MAX_VALUE"
