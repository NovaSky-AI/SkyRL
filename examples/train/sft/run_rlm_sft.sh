set -x

# SFT training for alphaXiv RLM agent trajectories on Qwen3.5-9B.
#
# export WANDB_API_KEY=<your_key>
# export HF_TOKEN=<your_token>
# bash examples/train/sft/run_rlm_sft.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
: "${UV_CACHE_DIR:=$PROJECT_ROOT/.uv-cache}"
: "${UV_PROJECT_ENVIRONMENT:=$PROJECT_ROOT/.venv}"
export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT


: "${LOGGER:=wandb}"
: "${NUM_GPUS:=8}"
: "${BATCH_SIZE:=16}"
: "${NUM_EPOCHS:=1}"
: "${MAX_LENGTH:=16384}"
: "${MICRO_BATCH_SIZE:=2}"
: "${LEARNING_RATE:=1e-5}"
: "${LOG_INTERVAL:=250}"
: "${WANDB_PROJECT:=alphaxiv-rlm-sft}"
: "${WANDB_RUN_NAME:=sft-qwen3.5-0.8b-multi-paper}"

export LOGGER NUM_GPUS BATCH_SIZE NUM_EPOCHS MAX_LENGTH MICRO_BATCH_SIZE LEARNING_RATE LOG_INTERVAL WANDB_PROJECT WANDB_RUN_NAME

uv run --extra fsdp python examples/train/sft/rlm_sft_trainer.py $@
