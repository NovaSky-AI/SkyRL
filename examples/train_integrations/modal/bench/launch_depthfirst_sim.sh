#!/bin/bash
# Launch the depthfirst-shaped DeepSeek-V4.1-Flash LoRA RL throughput run on Modal (16x B200 by default: 1 trainer
# node + 1 vLLM node; GPU=B300:8 for B300). Run from anywhere with the Modal CLI logged in:
#
#   bash examples/train_integrations/modal/bench/launch_depthfirst_sim.sh setup   # once: model, dataset, caches
#   bash examples/train_integrations/modal/bench/launch_depthfirst_sim.sh run     # the RL run (detached)
#
# Env: MODELS_VOLUME [dsv41-models] (Modal volume mounted at /models), MODAL_SECRETS [wandb; empty to log to the
# console only: then also LOGGER=console], GPU [B300:8], TIMEOUT [10800 s], and any run_depthfirst_sim.sh variable
# (MAX_TURNS, MAX_STALENESS_STEPS, MAX_TRAINING_STEPS, ...). Logs: /root/data/skyrl-logs/<RUN_NAME>/run.log on the
# skyrl-data volume, and the W&B project dsv41_long_horizon.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1  # examples/train_integrations/modal
: "${MODELS_VOLUME:=dsv41-models}"
: "${MODAL_SECRETS=wandb}"
: "${GPU:=B200:8}"
: "${TIMEOUT:=21600}"  # 69-turn rollouts: ~2 h of generation before the first step
: "${RUN_NAME:=dsv41_depthfirst_sim}"
B=examples/train_integrations/modal/bench
export MODAL_VOLUMES="$MODELS_VOLUME:/models" MODAL_UV_EXTRAS=megatron MODAL_SECRETS

case "${1:-run}" in
  setup)
    MODAL_NUM_NODES=1 MODAL_GPU=none MODAL_CPU=32 MODAL_MEMORY=262144 MODAL_TIMEOUT=14400 \
      modal run main.py --command "bash $B/setup_depthfirst_sim.sh"
    ;;
  run)
    PASS=""
    for v in MAX_TURNS MAX_MODEL_LEN TRAIN_BATCH_SIZE N_SAMPLES_PER_PROMPT MAX_STALENESS_STEPS MAX_TRAINING_STEPS \
             OFFLOAD_GB LOGGER GEN_STAGGER_S CP VLLM_DP; do
      [ -n "${!v:-}" ] && PASS="$PASS $v=${!v}"
    done
    CUBIN="/models/cache/flashinfer_cubin_${FLASHINFER_CUBIN_VERSION:-0.7.0.post1}"  # written by setup
    DSV41="SKYRL_DSV41_COMPRESS_FROZEN_WEIGHTS=1;SKYRL_DSV41_RECOMPUTE_DERIVE_MLP_INPUT=1;SKYRL_DSV41_RECOMPUTE_OFFLOAD=1"
    DSV41="$DSV41;SKYRL_DSV41_RECOMPUTE_OFFLOAD_HOST_GB=${OFFLOAD_GB:-24};SKYRL_DSV41_FAKE_QUANT=reference"
    MODAL_NUM_NODES=2 MODAL_GPU="$GPU" MODAL_TIMEOUT="$TIMEOUT" \
    MODAL_ENV="VLLM_CACHE_ROOT=/models/cache/${GPU%%:*}/vllm;VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/models/cache/${GPU%%:*}/flashinfer_autotune;FLASHINFER_WORKSPACE_BASE=/root/fi_ws;FLASHINFER_CUBIN_DIR=$CUBIN;FLASHINFER_NO_DOWNLOAD=1;MBRIDGE_STATE_REUSE_HANDLES=0;MBRIDGE_STATE_CACHE_SMALL_TENSORS=0;$DSV41" \
      modal run --detach main.py --command \
      "MODEL_PATH=/models/DeepSeek-V4.1-Flash-fp8 DATA_DIR=${DATA_DIR:-/root/data/gsm8k_df_t${MAX_TURNS:-69}} RUN_NAME=$RUN_NAME$PASS bash $B/run_depthfirst_sim.sh"
    ;;
  *) echo "usage: $0 setup|run" >&2; exit 2 ;;
esac
