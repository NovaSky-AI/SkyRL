set -x

# SFT training for alphaXiv page-relevance classification.
#
# export WANDB_API_KEY=<your_key>
# export HF_TOKEN=<your_token>
# bash examples/train/sft/run_sft.sh

: "${LOGGER:=wandb}"
: "${NUM_GPUS:=1}"
: "${BATCH_SIZE:=16}"
: "${NUM_STEPS:=4000}"
: "${MAX_LENGTH:=4096}"
: "${EVAL_INTERVAL:=512}"
: "${WANDB_PROJECT:=alphaxiv-page-labels}"
: "${WANDB_RUN_NAME:=sft-qwen3.5-2b-base}"

export LOGGER NUM_GPUS BATCH_SIZE NUM_STEPS MAX_LENGTH EVAL_INTERVAL WANDB_PROJECT WANDB_RUN_NAME

uv run --isolated --extra fsdp --with flash-linear-attention python examples/train/sft/sft_trainer.py $@
