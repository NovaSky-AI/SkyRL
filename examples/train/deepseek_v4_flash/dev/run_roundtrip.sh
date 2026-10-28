#!/usr/bin/env bash
# DeepSeek-V4-Flash Megatron <-> vLLM roundtrip tests on a single 8-GPU node, run directly (no Ray job):
#   bash examples/train/deepseek_v4_flash/dev/run_roundtrip.sh <pytest args> > $DSV4_ROOT/logs/rt-<id>.log 2>&1
# Models, caches and logs live under DSV4_ROOT (default ~/dsv4; point it at local NVMe); override any path via env.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
DSV4_ROOT=${DSV4_ROOT:-$HOME/dsv4}
# Isolated Ray per test run; never attach to a cluster already running on the node.
export RAY_ADDRESS=local
# Tests read ~/data/gsm8k.
mkdir -p ~/data
[ -e ~/data/gsm8k ] || ln -sfn "$DSV4_ROOT/data/gsm8k" ~/data/gsm8k
export SKYRL_DEEPSEEK_V4_FLASH_4LAYER=${SKYRL_DEEPSEEK_V4_FLASH_4LAYER:-$DSV4_ROOT/models/DeepSeek-V4-Flash-4layer}
export SKYRL_DEEPSEEK_V41_FLASH_6LAYER=${SKYRL_DEEPSEEK_V41_FLASH_6LAYER:-$DSV4_ROOT/models/DeepSeek-V4.1-Flash-6layer}
export SKYRL_DEEPSEEK_V41_FLASH_SUB_WIN_ENGRAM=${SKYRL_DEEPSEEK_V41_FLASH_SUB_WIN_ENGRAM:-$DSV4_ROOT/models/DeepSeek-V4.1-Flash-sub-win-engram}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$DSV4_ROOT/uv_cache}
export HF_HOME=${HF_HOME:-$DSV4_ROOT/hf_cache}
RUN_ID=${RUN_ID:-$(date +%s)}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$DSV4_ROOT/vllm_cache/shared}
# Fresh FlashInfer autotune cache per run: with a warm cache, vLLM boot hung with TP0 done and TP1-3 stuck in
# the collective `trtllm::fused_moe::gemm1` tuning; cold-cache boots were fine.
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-$DSV4_ROOT/vllm_cache/autotune/$RUN_ID}
# TileLang JIT (vLLM's mHC kernels) must use the system toolkit. Without /usr/local/cuda/bin on PATH (any
# non-interactive shell) it falls back to the venv's nvidia/cu13 nvcc, which rejects the venv's CCCL headers:
# "CUDA compiler and CUDA toolkit headers are incompatible".
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
# Checkpoint tests write into a mkdtemp; a full-FT slice checkpoint (fp32 main params + optimizer state) can fill
# a small root disk's /tmp.
export SKYRL_TEST_CKPT_DIR=${SKYRL_TEST_CKPT_DIR:-$DSV4_ROOT/ckpt-tests}
mkdir -p "$SKYRL_TEST_CKPT_DIR"
# The default /tmp/skyrl-logs may belong to another user on a shared node.
export SKYRL_TEST_LOG_PATH=${SKYRL_TEST_LOG_PATH:-$DSV4_ROOT/logs/skyrl-infra}
# GCP A4 images force Google's gIB NCCL net plugin (/usr/local/gib, built for NCCL 2.27.5); the venv's NCCL 2.29.7
# fails "Failed to initialize any NET plugin" -> ncclInvalidUsage at vLLM init. Single node needs no net plugin.
unset NCCL_NET NCCL_TUNER_CONFIG_PATH
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/usr/local/gib' | paste -sd: -)
echo "RUN_ID=$RUN_ID VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR SKYRL_DEEPSEEK_V4_FLASH_4LAYER=$SKYRL_DEEPSEEK_V4_FLASH_4LAYER"
uv run --isolated --extra megatron --extra dev python -m pytest -s -v -m "h100 or not h100" "$@"
