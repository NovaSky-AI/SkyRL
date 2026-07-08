#!/usr/bin/env bash

# Tinker API server (Megatron backend) for DPPO LoRA training on Qwen3.6-35B-A3B.
#
# Mirrors the training-side setup of
# examples/train/megatron/run_megatron_dapo_qwen3.6_35b_a3b_lora.sh (Megatron
# TP=4 / PP=1 / CP=1 / EP=8 / ETP=1, LoRA rank 32, colocated vLLM, nccl weight
# sync, language_model_only, GDN triton prefill), scaled to a single node of 8
# GPUs by default. Deviations from that script:
#   - merge_lora=false: the Tinker sampling path routes requests by LoRA
#     adapter name on vLLM, which requires adapter (not merged-weight) sync.
#   - CPU optimizer offload is off: LoRA-only optimizer state is tiny.
#   - Loss/algorithm knobs (DPPO deltas, advantages, batch sizes, LR) are
#     driven by the client per request; only `dppo_type` is server-side.
#
#   bash examples/tinker/dppo/run_tinker_server_megatron.sh
#
# Then run the client: examples/tinker/dppo/dppo_client.py

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
PORT="${PORT:-8000}"
NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-1}"
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE="${INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE:-8}"

# megatron config (matches the DAPO example)
MEGATRON_TP="${MEGATRON_TP:-4}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_EP="${MEGATRON_EP:-8}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"

# DPPO divergence variant: "binary_tv" or "binary_kl". Per-request thresholds
# (delta_low/delta_high) are passed by the client via loss_fn_config.
DPPO_TYPE="${DPPO_TYPE:-binary_tv}"

# Qwen3.6 flags (see the DAPO example for background)
ENFORCE_EAGER=false # cudagraphs need FULL_DECODE_ONLY below
LANGUAGE_MODEL_ONLY=true
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

DATABASE_URL="${DATABASE_URL:-sqlite:///$HOME/.cache/skyrl_tinker_dppo/tinker.db}"
CHECKPOINTS_BASE="${CHECKPOINTS_BASE:-$HOME/.cache/skyrl_tinker_dppo/checkpoints}"
mkdir -p "$(dirname "${DATABASE_URL#sqlite:///}")" "$CHECKPOINTS_BASE"

BACKEND_CONFIG=$(cat <<EOF
{
  "trainer.placement.colocate_all": true,
  "trainer.placement.policy_num_nodes": $NUM_NODES,
  "trainer.placement.policy_num_gpus_per_node": $NUM_GPUS_PER_NODE,
  "trainer.policy.megatron_config.tensor_model_parallel_size": $MEGATRON_TP,
  "trainer.policy.megatron_config.pipeline_model_parallel_size": $MEGATRON_PP,
  "trainer.policy.megatron_config.context_parallel_size": $MEGATRON_CP,
  "trainer.policy.megatron_config.expert_model_parallel_size": $MEGATRON_EP,
  "trainer.policy.megatron_config.expert_tensor_parallel_size": $MEGATRON_ETP,
  "trainer.policy.megatron_config.lora_config.merge_lora": false,
  "trainer.policy.language_model_only": $LANGUAGE_MODEL_ONLY,
  "trainer.micro_train_batch_size_per_gpu": 1,
  "trainer.micro_forward_batch_size_per_gpu": 1,
  "trainer.policy.optimizer_config.weight_decay": 0.1,
  "trainer.policy.optimizer_config.max_grad_norm": 1.0,
  "trainer.algorithm.dppo.dppo_type": "$DPPO_TYPE",
  "generator.inference_engine.num_engines": $NUM_INFERENCE_ENGINES,
  "generator.inference_engine.tensor_parallel_size": $INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE,
  "generator.inference_engine.backend": "vllm",
  "generator.inference_engine.run_engines_locally": true,
  "generator.inference_engine.weight_sync_backend": "nccl",
  "generator.inference_engine.gpu_memory_utilization": 0.7,
  "generator.inference_engine.enforce_eager": $ENFORCE_EAGER,
  "generator.inference_engine.distributed_executor_backend": "mp",
  "generator.inference_engine.language_model_only": $LANGUAGE_MODEL_ONLY,
  "generator.inference_engine.engine_init_kwargs": {"gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}
}
EOF
)

uv run --isolated --extra tinker --extra megatron -m skyrl.tinker.api \
  --base-model "$MODEL_NAME" \
  --backend megatron \
  --host 0.0.0.0 \
  --port "$PORT" \
  --database-url "$DATABASE_URL" \
  --checkpoints-base "$CHECKPOINTS_BASE" \
  --backend-config "$BACKEND_CONFIG" \
  "$@"
