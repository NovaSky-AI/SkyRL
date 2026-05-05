#!/usr/bin/env bash
# Download FleetAI/fleet-cu-trajectories dataset from HuggingFace Hub.
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/fleet-download-trajectories.sh
#   # or: huggingface-cli login  then  bash scripts/fleet-download-trajectories.sh
set -eo pipefail

cd "$(dirname "$0")/.."

pip install -q huggingface_hub

python3 download.py
