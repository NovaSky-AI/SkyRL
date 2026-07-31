#!/usr/bin/env bash
set -euo pipefail

# Unique per invocation (seconds + PID): the shared wandb project means an hour-granular
# name can collide with a concurrent run on another host, and get_summary.py would then
# read the wrong run.
RUN_NAME="run_$(date +%Y%m%d%H%M%S)_$$"

SCRIPT_DIR=$(dirname $(realpath $0))
# Thresholds: 5% allowance from min/max of the last 5 CI runs as of 31st July 2026
# (eval min 0.5292, train reward min 0.2141).
#
# Rebaselined from the original 23rd Feb 2026 values (eval 0.56, train 0.32). Three
# intentional changes to the fully-async recipe moved these metrics between then and now,
# and the gate was never updated -- `loss/avg_final_rewards` had been failing on every
# nightly run since 30th June:
#
#   #1798 (18 Jun)  stopped clearing the KV cache on weight sync, so rollouts reused KV
#                   computed under stale weights (vllm/prefix_cache_hit_rate 0.15 -> 0.92).
#                   eval 0.59 -> 0.50, train reward 0.32 -> 0.29.
#   #1850 (30 Jun)  switched the recipe to policy_loss_type=rollout_is. Before this the
#                   default `regular` loss recomputed old logprobs, so the IS ratio was
#                   identically 1 at update_epochs_per_batch=1 -- clip_ratio was exactly
#                   0.0 and there was no importance-sampling correction at all despite
#                   max_staleness_steps=4. Correcting it halved grad_norm (0.28 -> 0.16)
#                   and is the dominant cause of the lower reward: train 0.29 -> 0.24.
#   #1836 (14 Jul)  per-global_step KV cache salt, reversing #1798's stale-KV reuse
#                   (prefix_cache_hit_rate back to 0.15). eval 0.41 -> 0.50.
#   #1929 (20 Jul)  logged eval metrics after eval instead of before, so the summary now
#                   holds the final-step eval rather than the step-8 one. eval 0.50 -> 0.58.
#
# eval/all/avg_score has recovered to its pre-June level, but it sits close enough to the
# old 0.56 gate that 3 of the last 22 runs fell below it, so it is rebaselined too.
# avg_num_tokens and the logprobs diff were both comfortably inside their gates throughout
# and are left as-is -- as in Feb, they are deliberately loose guardrails rather than tight
# regression gates (the last 5 runs sit at 261-268 tokens and 0.0167-0.0174).
EVAL_ACC_MIN_VALUE=0.50
TRAIN_ACC_MIN_VALUE=0.20
AVG_NUM_TOKENS_MAX_VALUE=283
LOGPROBS_DIFF_MAX_VALUE=0.040

# The anyscale job's working_dir is the repo root, so we can use relative paths.
bash examples/train/fully_async/fully_async_run_gsm8k.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.run_name=\"$RUN_NAME\" trainer.project_name=\"gsm8k_fully_async_ci\"

uv run --isolated --extra fsdp $SCRIPT_DIR/get_summary.py --run_name $RUN_NAME --project_name "gsm8k_fully_async_ci" --asserts "eval/all/avg_score >= $EVAL_ACC_MIN_VALUE" "loss/avg_final_rewards >= $TRAIN_ACC_MIN_VALUE" "generate/avg_num_tokens <= $AVG_NUM_TOKENS_MAX_VALUE" "policy/rollout_train_logprobs_abs_diff_mean <= $LOGPROBS_DIFF_MAX_VALUE"
