#!/usr/bin/env bash
# DeepSeek-V4-Flash Megatron <-> vLLM roundtrip tests on a worker node, submitted as a Ray job:
#   ray job submit --entrypoint-num-gpus 8 -- bash examples/train/deepseek_v4_flash/dev/run_roundtrip.sh <pytest args>
# Run from a copy of the repo on /shared (worker nodes can't see the head node's disk).
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export RAY_ADDRESS=local
# The job holds the whole node; a local raylet rejects a CUDA_VISIBLE_DEVICES subset of the node GPUs.
unset CUDA_VISIBLE_DEVICES
# Anyscale turns on event export to its aggregator; a nested local Ray has none and SEGVs on shutdown.
export RAY_enable_core_worker_ray_event_to_aggregator=0 RAY_enable_ray_event=0
# The outer Ray job's runtime env leaks into nested ray.init calls (vLLM's engine core); merge instead of failing.
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
# Tests read ~/data/gsm8k; worker nodes only have it on /shared.
mkdir -p ~/data && ln -sfn /shared/hackskyrl/data/gsm8k ~/data/gsm8k
export SKYRL_DEEPSEEK_V4_FLASH_4LAYER=${SKYRL_DEEPSEEK_V4_FLASH_4LAYER:-/shared/hackskyrl/dsv4_flash/DeepSeek-V4-Flash-4layer}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/shared/hackskyrl/uv_cache}
uv run --isolated --extra megatron --extra dev python -m pytest -s -v -m "h100 or not h100" "$@"
