#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CKPT_DIR="${1:-/root/neer/runs/019d4f91-0c00-723a-ba20-fefa840d3024/.neer/artifacts/step_480}"
HF_REPO="${2:-alphaXiv/rlm-sft-multi-9b-v1}"
DTYPE="${3:-bfloat16}"

OUTPUT_DIR="${CKPT_DIR}/hf_export"

echo "=== FSDP → HuggingFace conversion ==="
echo "  Checkpoint : ${CKPT_DIR}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Dtype      : ${DTYPE}"
echo ""

cd "$REPO_ROOT"
uv run python scripts/convert_fsdp_to_hf.py \
    --ckpt-dir "$CKPT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --dtype "$DTYPE"

echo ""
echo "=== Uploading to HuggingFace: ${HF_REPO} ==="
uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('${HF_REPO}', exist_ok=True)
api.upload_folder(
    folder_path='${OUTPUT_DIR}',
    repo_id='${HF_REPO}',
    commit_message='Upload model weights from step_480 checkpoint',
)
print('Upload complete: https://huggingface.co/${HF_REPO}')
"
