#!/bin/bash
# One-time prep on the Modal volume (/root/data) for the DeepSeek-V4.1-Flash recipes. CPU only:
#   modal run examples/train_integrations/modal/main_multinode.py::prep \
#     --command "HF_TOKEN=... bash examples/train/deepseek_v4_flash/modal/prepare_dsv41_modal.sh"
#
# 1. downloads the FP8/MXFP4 release (476 GB) to $RELEASE_DIR,
# 2. builds the BF16 text-only copy the recipe loads (1.2 TB for all 40 layers; LAYERS=0,1,2,3,20,21,22,23,24 for the
#    9-layer slice with every V4.1 layer role, ~330 GB) with the chat-mode template,
# 3. downloads and preprocesses the DAPO data.
# Each step is skipped when its output already exists.
set -euxo pipefail
HF_REPO=${HF_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}   # check the revision matches the -1017 release you want
RELEASE_DIR=${RELEASE_DIR:-/root/data/models/DeepSeek-V4.1-Flash-1017}
LAYERS=${LAYERS:-}                                   # empty = all layers
OUT=${OUT:-/root/data/models/DeepSeek-V4.1-Flash-bf16-chat${LAYERS:+-l$(echo $LAYERS | tr ',' '_')}}
WORKERS=${WORKERS:-32}

if [ ! -f "$RELEASE_DIR/config.json" ]; then
  uvx --from "huggingface_hub[hf_transfer]" hf download "$HF_REPO" --local-dir "$RELEASE_DIR"
fi
if [ ! -f "$OUT/model.safetensors.index.json" ]; then
  DSV41_LOCAL_DIR="$RELEASE_DIR" uv run --no-project --with torch --with safetensors --with numpy \
    python examples/train/deepseek_v4_flash/dev/convert_dsv41_bf16_parallel.py "$OUT" "$LAYERS" "$WORKERS"
fi
if [ ! -f /root/data/dapo/dapo-math-17k-cleaned.parquet ]; then
  DATA_DIR=/root/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
fi
echo "model: $OUT"
