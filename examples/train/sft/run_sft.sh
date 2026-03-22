set -x

# SFT training for alphaXiv page-relevance classification.
#
# export WANDB_API_KEY=<your_key>
# export HF_TOKEN=<your_token>
# bash examples/train/sft/run_sft.sh

: "${LOGGER:=wandb}"
: "${NUM_GPUS:=1}"
: "${BATCH_SIZE:=4}"
: "${NUM_STEPS:=500}"
: "${MAX_LENGTH:=2048}"
: "${EVAL_INTERVAL:=50}"
: "${WANDB_PROJECT:=alphaxiv-page-labels}"
: "${WANDB_RUN_NAME:=sft-qwen2.5-0.5b}"

export LOGGER NUM_GPUS BATCH_SIZE NUM_STEPS MAX_LENGTH EVAL_INTERVAL WANDB_PROJECT WANDB_RUN_NAME

uv run --isolated --extra fsdp python examples/train/sft/sft_trainer.py $@
