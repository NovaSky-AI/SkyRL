set -x

# SFT training with Megatron backend for Qwen3-0.6B
#
# This script runs supervised fine-tuning using Megatron's tensor and pipeline
# parallelism (TP=2, PP=2) on 4 GPUs with the Alpaca dataset.
# All configuration is handled inside the Python script.
#
# Usage:
#   bash examples/train/sft/run_sft_megatron.sh

uv run --isolated --extra megatron python examples/train/sft/sft_megatron_trainer.py
