#!/usr/bin/env bash
# Similar to https://github.com/volcengine/verl/blob/recipe/dapo/recipe/dapo/prepare_dapo_data.sh
# Example: DATA_DIR=/path/to/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
set -euxo pipefail

export DATA_DIR=${DATA_DIR:-"${HOME}/data/dapo"}
export TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${DATA_DIR}/aime-2024.parquet"}
export TEST_FILE_2025=${TEST_FILE_2025:-"${DATA_DIR}/aime-2025.jsonl"}

mkdir -p "${DATA_DIR}"

wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"

wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"

# AIME 2025 (I + II, 30 problems), pinned revision. Optional eval set, written to aime-2025-cleaned.parquet.
wget -O "${TEST_FILE_2025}" "https://huggingface.co/datasets/zhuzilin/aime-2025/resolve/3d2ffb246285920641112d03a9c23218db4de9b7/aime-2025.jsonl?download=true"

# remove duplicate rows in the original dataset
# https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/discussions/3 
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.data_preprocess_dapo_aime --data-dir "${DATA_DIR}"

# convert AIME 2025 to the AIME-2024 prompt / reward format
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.data_preprocess_aime2025 --data-dir "${DATA_DIR}" --input-file "${TEST_FILE_2025}"
