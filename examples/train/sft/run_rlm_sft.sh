set -x

# SFT training for alphaXiv RLM agent trajectories on Qwen3.5-9B.
#
# export WANDB_API_KEY=<your_key>
# export HF_TOKEN=<your_token>
# bash examples/train/sft/run_rlm_sft.sh

: "${LOGGER:=wandb}"
: "${NUM_GPUS:=4}"
: "${BATCH_SIZE:=4}"
: "${NUM_EPOCHS:=1}"
: "${MAX_LENGTH:=16384}"
: "${MICRO_BATCH_SIZE:=1}"
: "${LEARNING_RATE:=1e-5}"
: "${LOG_INTERVAL:=10}"
: "${WANDB_PROJECT:=alphaxiv-rlm-sft}"
: "${WANDB_RUN_NAME:=sft-qwen3.5-9b-VLM}"

export LOGGER NUM_GPUS BATCH_SIZE NUM_EPOCHS MAX_LENGTH MICRO_BATCH_SIZE LEARNING_RATE LOG_INTERVAL WANDB_PROJECT WANDB_RUN_NAME

uv run --isolated --extra fsdp python examples/train/sft/rlm_sft_trainer.py $@
