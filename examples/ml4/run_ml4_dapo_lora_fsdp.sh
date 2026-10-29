#!/usr/bin/env bash
# ML4 GRPO on DAPO-Math-17K with AIME-2024 validation: FSDP2 + PEFT LoRA, colocated vLLM.
# Recipe values live in ml4_dapo_lora_fsdp.yaml; this script adds the paths and any extra
# overrides passed as arguments (key=value, applied last).
#
# Prerequisites:
#   - SkyRL installed into the image's Python env (vLLM from the image), Ray cluster up on all nodes.
#   - ML4_MODEL_DIR written by `python -m ml4_train.convert <ckpt> <dir>` (config.json with auto_map,
#     model code, links to the consolidated shards and tekken.json). The trainer tokenizer is
#     transformers' MistralCommonBackend over tekken.json (same library as vLLM's tokenizer_mode=mistral);
#     this script adds the tokenizer_config.json that selects it. Verify once with
#     `python examples/ml4/check_tokenizer.py $ML4_MODEL_DIR --parquet $DATA_DIR/dapo-math-17k-cleaned.parquet`.
#   - Data: DATA_DIR=... bash examples/train/algorithms/dapo/prepare_dapo_data.sh (writes *-cleaned.parquet;
#     AIME-2024 deduplicated to 30 problems).
#
#   ML4_MODEL_DIR=/path/to/ml4-hf DATA_DIR=/path/to/dapo CKPT_DIR=/node-local/ckpts \
#     bash examples/ml4/run_ml4_dapo_lora_fsdp.sh [key=value ...]
set -euo pipefail

: "${ML4_MODEL_DIR:?set ML4_MODEL_DIR (output of ml4_train.convert)}"
: "${DATA_DIR:?set DATA_DIR (output of prepare_dapo_data.sh)}"
CKPT_DIR=${CKPT_DIR:-"$HOME/ckpts/ml4_dapo_lora_fsdp"}
EXPORT_DIR=${EXPORT_DIR:-"$HOME/exports/ml4_dapo_lora_fsdp"}
PYTHON=${PYTHON:-python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Opt-in (ML4_LORA_SPLITK1=1): LoRA shrink with split_k=1 ($HERE/lora_tuned/NVIDIA_GB200_SHRINK.json) for
# bitwise-repeatable rollout. Default is vLLM's split_k=64/8: correct, but fp32 atomics make the last bits vary.
# VLLM_TUNED_CONFIG_FOLDER is forwarded to the engines by prepare_runtime_environment.
if [[ "${ML4_LORA_SPLITK1:-0}" != 0 ]]; then
  export VLLM_TUNED_CONFIG_FOLDER=${VLLM_TUNED_CONFIG_FOLDER:-"$HERE/lora_tuned"}
fi

# Without it AutoTokenizer does not pick mistral-common for this model type (or converts tekken.json
# into a TokenizersBackend with different ids).
TOK_CFG="$ML4_MODEL_DIR/tokenizer_config.json"
[[ -f "$TOK_CFG" ]] || echo '{"tokenizer_class": "MistralCommonBackend"}' > "$TOK_CFG"
grep -q MistralCommonBackend "$TOK_CFG" || { echo "$TOK_CFG must select MistralCommonBackend" >&2; exit 1; }

# yaml -> SkyRL dotlist (same serialization SkyRLTrainConfig.from_cli_overrides uses for dicts)
mapfile -t RECIPE < <("$PYTHON" -c '
import sys, yaml
from skyrl.train.config.config import overrides_dict_to_dotlist
print("\n".join(overrides_dict_to_dotlist(yaml.safe_load(open(sys.argv[1])))))
' "$HERE/ml4_dapo_lora_fsdp.yaml")

"$PYTHON" -m skyrl.train.entrypoints.main_base \
  "${RECIPE[@]}" \
  trainer.policy.model.path="$ML4_MODEL_DIR" \
  data.train_data="['$DATA_DIR/dapo-math-17k-cleaned.parquet']" \
  data.val_data="['$DATA_DIR/aime-2024-cleaned.parquet']" \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.export_path="$EXPORT_DIR" \
  "$@"
