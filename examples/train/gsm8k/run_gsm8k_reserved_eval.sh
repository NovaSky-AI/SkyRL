set -x

# GSM8K GRPO with evaluation on a reserved engine group.
#
# Training and its inference engines stay colocated on NUM_GPUS GPUs (run_gsm8k.sh); evaluation
# runs on NUM_EVAL_GPUS additional, eval-only engines that never join weight sync. Each eval loads
# the HF export written at the same step, so the training loop only admits the eval and moves on;
# results land at the step they measured, with `eval/lag_steps` saying how far the loop had moved.
#
# `trainer.eval_interval` must be a multiple of `trainer.hf_save_interval` (the export is the
# snapshot), and `trainer.max_hf_exports_to_keep` bounds the disk those exports take: an export a
# queued eval still needs is never deleted.
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# NUM_GPUS=4 NUM_EVAL_GPUS=1 bash examples/train/gsm8k/run_gsm8k_reserved_eval.sh

: "${NUM_GPUS:=4}"
: "${NUM_EVAL_GPUS:=1}"

SCRIPT_DIR=$(dirname "$(realpath "$0")")

NUM_GPUS=$NUM_GPUS bash "$SCRIPT_DIR/run_gsm8k.sh" \
  trainer.eval_dispatch.mode=reserved \
  trainer.eval_dispatch.num_engines=$NUM_EVAL_GPUS \
  trainer.eval_dispatch.max_queue_size=2 \
  trainer.eval_dispatch.overflow_policy=backpressure \
  trainer.eval_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.max_hf_exports_to_keep=2 \
  trainer.export_path="$HOME/exports/gsm8k_1.5B_reserved_eval" \
  trainer.run_name="gsm8k_reserved_eval" \
  "$@"
