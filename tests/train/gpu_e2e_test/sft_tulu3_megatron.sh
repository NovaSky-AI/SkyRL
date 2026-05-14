#!/usr/bin/env bash
# E2E CI test for SFT with the Megatron backend on Tulu3.
#
# Runs ``examples/train/sft/run_sft_megatron_tulu3_50k.sh`` with shorter
# overrides (100 steps, train[:2000]) and asserts:
#   * Via ``check_sft_trend.py`` (sourcing the run's history from wandb):
#       - The run completed all expected steps.
#       - No NaN/inf in the ``train/loss`` history.
#       - Mean of the last 5 logged losses is strictly less than the mean of the
#         first 5 (lenient trend check averaged over windows to absorb step
#         noise; no magnitude thresholds).
#
# Logger is wandb so that the run is visible alongside other CI runs and
# downstream assertions can introspect the run's logged history directly.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUN_NAME="sft_megatron_run_$(date +%Y%m%d%H)"
PROJECT_NAME="skyrl_sft_ci"
ENTITY="sky-posttraining-uc-berkeley"
NUM_STEPS=100
LOG_FILE="${LOG_FILE:-/tmp/${RUN_NAME}.log}"

# The anyscale job's working_dir is the repo root, so we can use relative paths.
# We pipe through `tee` so the full stdout is mirrored to ``$LOG_FILE`` for
# downstream parsing of the loss trend / completion signal.
#
# Notes on overrides vs the source script:
#   * lr is bumped from 1e-6 to 1e-4 so the model produces a clear downward
#     trend in 100 steps; the source script's 1e-6 is calibrated for 4166 steps.
#   * batch_size=8, micro_train_batch_size_per_gpu=2 are sized for L4_ci (4 GPUs).
bash examples/train/sft/run_sft_megatron_tulu3_50k.sh \
  num_steps=$NUM_STEPS \
  dataset_split="train[:2000]" \
  batch_size=8 \
  micro_train_batch_size_per_gpu=2 \
  max_length=1024 \
  model.path=Qwen/Qwen2.5-0.5B-Instruct \
  optimizer_config.lr=1e-4 \
  placement.num_nodes=1 \
  placement.num_gpus_per_node=4 \
  megatron_config.tensor_model_parallel_size=1 \
  megatron_config.pipeline_model_parallel_size=1 \
  megatron_config.context_parallel_size=1 \
  train_on_what="all_assistant_messages" \
  logger=wandb \
  project_name="$PROJECT_NAME" \
  run_name="$RUN_NAME" \
  ckpt_path="" \
  ckpt_interval=0 \
  hf_save_interval=0 \
  resume_from="" \
  2>&1 | tee "$LOG_FILE"


# ---- Wandb-side assertions ----
# Pulls the run's logged ``train/loss`` history and asserts:
#   * final _step >= NUM_STEPS (completion),
#   * no NaN/inf in the history,
#   * mean(last 5) < mean(first 5) (lenient windowed trend).
uv run --isolated --extra fsdp $SCRIPT_DIR/check_sft_trend.py \
  --run_name "$RUN_NAME" \
  --project_name "$PROJECT_NAME" \
  --entity "$ENTITY" \
  --window 5 \
  --expected_steps "$NUM_STEPS"

echo "All SFT CI assertions passed."
