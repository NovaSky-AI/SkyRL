#!/usr/bin/env bash
set -euo pipefail

# GSM8K GRPO with evaluation on a reserved engine group: colocated training + its engines on
# NUM_GPUS GPUs, plus one eval-only engine on its own GPU. Evals load the HF export written at the
# same step, so the loop never waits for them. Needs NUM_GPUS + 1 GPUs (3 by default).
#
# Include seconds + PID so the name is unique per invocation (see gsm8k_colocate.sh).
RUN_NAME="run_$(date +%Y%m%d%H%M%S)_$$"
SCRIPT_DIR=$(dirname $(realpath $0))
EXPORT_PATH="/tmp/skyrl-exports/$RUN_NAME"

# Same eval/train thresholds as gsm8k_colocate.sh: a reserved eval measures the same checkpoints.
EVAL_ACC_MIN_VALUE=0.69
TRAIN_ACC_MIN_VALUE=0.69

# Two training GPUs (dp=2 keeps the batch math of run_gsm8k.sh valid) and one reserved eval engine.
NUM_GPUS=2 NUM_EVAL_GPUS=1 bash examples/train/gsm8k/run_gsm8k_reserved_eval.sh \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.export_path="$EXPORT_PATH" \
  trainer.project_name=\"gsm8k_ci\" \
  trainer.run_name=\"$RUN_NAME\"

# The summary holds each metric's last value: the final-step eval (forced, with its own export),
# its lag, an eval site that only admits (no stall), and no skipped points of any kind.
uv run --isolated --extra fsdp $SCRIPT_DIR/get_summary.py --run_name $RUN_NAME --project_name "gsm8k_ci" \
  --asserts "eval/all/avg_score >= $EVAL_ACC_MIN_VALUE" "loss/avg_final_rewards >= $TRAIN_ACC_MIN_VALUE" \
            "eval/lag_steps >= 0" "timing/eval <= 5" \
  --absent eval/skipped_busy eval/skipped_crashed eval/skipped_cancelled \
           eval/skipped_ckpt_missing eval/skipped_sync_failed eval/skipped_sync_mismatch

# Retention: max_hf_exports_to_keep=2 in the example, so at most two exports remain (plus none of
# the ref-sync scratch tree).
NUM_EXPORTS=$(ls -d "$EXPORT_PATH"/global_step_* | wc -l)
if [ "$NUM_EXPORTS" -gt 2 ]; then
  echo "expected at most 2 HF exports under $EXPORT_PATH, found $NUM_EXPORTS"; exit 1
fi
echo "Retention check passed: $NUM_EXPORTS export(s) kept"
