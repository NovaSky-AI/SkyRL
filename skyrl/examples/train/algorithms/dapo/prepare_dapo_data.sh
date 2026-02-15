#!/usr/bin/env bash
# Similar to https://github.com/volcengine/verl/blob/recipe/dapo/recipe/dapo/prepare_dapo_data.sh
# Run from `skyrl-train` directory.
# Example: DATA_DIR=/path/to/data/dapo bash examples/algorithms/dapo/prepare_dapo_data.sh
set -uxo pipefail

export DATA_DIR=${DATA_DIR:-"${HOME}/data/dapo"}
export TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${DATA_DIR}/aime-2024.parquet"}

mkdir -p "${DATA_DIR}"

wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"

wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"

# remove duplicate rows in the original dataset
# https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/discussions/3 
uv run --isolated --extra vllm -m examples.algorithms.dapo.data_preprocess_dapo_aime --data-dir ${DATA_DIR}