#!/bin/bash
# One-time setup for run_dsv41_long_horizon.sh (CPU only; the model dir and DATA_DIR must be visible to every node):
#   1. download deepseek-ai/DeepSeek-V4.1-Flash (released FP8 / MXFP4, ~480 GB) and make it text-only in place
#      (drops the vision tower and the DSpark / MTP stages; weights unchanged),
#   2. build the long-horizon GSM8k dataset (MAX_TURNS turns of ~OBS_TOKENS-token tool observations).
#
#   MODEL_PATH=/mnt/shared/DeepSeek-V4.1-Flash DATA_DIR=/mnt/shared/data/gsm8k_long_horizon_t69 \
#     bash examples/train/deepseek_v4_flash/long_horizon/setup_dsv41_long_horizon.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."
: "${MODEL_PATH:?where to put the model (shared storage)}"
: "${MAX_TURNS:=69}"
: "${DATA_DIR:=$HOME/data/gsm8k_long_horizon_t$MAX_TURNS}"
: "${OBS_TOKENS:=1870}"
export MODEL_PATH DATA_DIR MAX_TURNS OBS_TOKENS
uv run --isolated --extra megatron bash -c '
set -euo pipefail
if [ ! -f "$MODEL_PATH/.text_only_done" ]; then
  python -c "from huggingface_hub import snapshot_download; snapshot_download(\"deepseek-ai/DeepSeek-V4.1-Flash\", local_dir=\"$MODEL_PATH\", max_workers=32)"
  python examples/train/deepseek_v4_flash/make_text_only_checkpoint.py "$MODEL_PATH" "$MODEL_PATH"
  touch "$MODEL_PATH/.text_only_done"
fi
python examples/train/deepseek_v4_flash/long_horizon_dataset.py --output_dir "$DATA_DIR" \
  --max-turns "$MAX_TURNS" --obs-tokens "$OBS_TOKENS"
'
echo "setup done: MODEL_PATH=$MODEL_PATH DATA_DIR=$DATA_DIR"
