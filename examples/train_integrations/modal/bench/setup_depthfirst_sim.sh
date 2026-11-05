#!/bin/bash
# One-time setup for run_depthfirst_sim.sh, run inside a Modal container (see launch_depthfirst_sim.sh setup):
#   1. download deepseek-ai/DeepSeek-V4.1-Flash (FP8 / MXFP4, ~480 GB) to the models volume and make it text-only
#      in place (drops the vision tower and the DSpark stages; weights unchanged),
#   2. build the long-horizon GSM8k dataset on the data volume,
#   3. copy FlashInfer's cubin / header package to the models volume. On Modal every Ray worker gets its own
#      isolated uv environment, so each process would otherwise see a different FLASHINFER_CUBIN_DIR and the TRT-LLM
#      MoE kernel's header symlinks would race ("Header KernelTraits.h not found"); the run points them all here.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
: "${MODELS_DIR:=/models}"
: "${MODEL_DIR:=$MODELS_DIR/DeepSeek-V4.1-Flash-fp8}"
: "${DATA_DIR:=/root/data/gsm8k_df_t16}"
: "${MAX_TURNS:=16}"
: "${OBS_TOKENS:=1870}"
export MODEL_DIR DATA_DIR MAX_TURNS OBS_TOKENS MODELS_DIR
uv run --isolated --extra megatron bash -c '
set -euo pipefail
if [ ! -f "$MODEL_DIR/.text_only_done" ]; then
  python -c "from huggingface_hub import snapshot_download; snapshot_download(\"deepseek-ai/DeepSeek-V4.1-Flash\", local_dir=\"$MODEL_DIR\", max_workers=32)"
  python examples/train/deepseek_v4_flash/make_text_only_checkpoint.py "$MODEL_DIR" "$MODEL_DIR"
  touch "$MODEL_DIR/.text_only_done"
fi
python examples/train/deepseek_v4_flash/long_horizon_dataset.py --output_dir "$DATA_DIR" \
  --max-turns "$MAX_TURNS" --obs-tokens "$OBS_TOKENS"
python - <<EOF
import os, shutil, flashinfer_cubin
from importlib.metadata import version
src = os.path.join(os.path.dirname(flashinfer_cubin.__file__), "cubins")
dst = os.path.join(os.environ["MODELS_DIR"], "cache", "flashinfer_cubin_" + version("flashinfer-cubin"))
if not os.path.isdir(dst):
    shutil.copytree(src, dst)
print("FLASHINFER_CUBIN_DIR=" + dst)
EOF
'
echo "setup done: MODEL_PATH=$MODEL_DIR DATA_DIR=$DATA_DIR"
