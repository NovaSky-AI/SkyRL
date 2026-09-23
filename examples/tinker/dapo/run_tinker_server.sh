#!/usr/bin/env bash
# Launch the SkyRL Tinker API server for the DAPO recipes on Qwen/Qwen3-30B-A3B-Base (Megatron backend).
#
# Mirrors the server-side settings of
#   examples/train/algorithms/dapo/run_dapo_qwen3_30b_a3b_lora_megatron_aime.sh  (LoRA, default)
#   examples/train/algorithms/dapo/run_dapo_qwen3_30b_a3b_megatron_aime.sh       (full fine-tuning, FULL_FT=1)
# on the reference layout: 2 nodes of 8xH100, Megatron TP4/EP8, 2 vLLM engines at TP=8 (one per node).
#
# The algorithm knobs a Tinker client cannot express live here in backend_config:
#   - dual-clip loss type + clip_ratio_c (the client sends loss_fn="ppo" with the clip thresholds)
#   - token_mean_legacy loss reduction, weight decay, grad clipping
#   - off-policy correction: geometric sequence masking (NOT TIS). Needs the client to send
#     `rollout_logprobs` (dapo_client.py does this when RECOMPUTE_OLD_LOGPROBS=1).
#
# Usage:
#   bash examples/tinker/dapo/run_tinker_server.sh                 # LoRA recipe (rank/alpha 128; client --lora-rank 128, LR 1e-5)
#   FULL_FT=1 bash examples/tinker/dapo/run_tinker_server.sh       # full fine-tuning recipe (client --lora-rank 0, LR 1e-6)
# The learning rate lives on the CLIENT (dapo_client.py picks 1e-5 or 1e-6 from --lora-rank); pair FULL_FT=1
# with --lora-rank 0 and vice versa.
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-30B-A3B-Base}"
PORT="${PORT:-8000}"
FULL_FT="${FULL_FT:-0}"

NUM_NODES="${NUM_NODES:-2}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-2}"
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE="${INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE:-8}"
# The native reference scripts use 0.7. On the Tinker path the trainer keeps ~25 GiB/GPU resident after
# offload once optimizer state exists (measured on the 30B LoRA recipe, 16xH100), while vLLM sizes its KV
# cache against the ~13 GiB seen at engine init; 0.7 then OOMs in sampling on the second full step. 0.6 leaves
# the headroom (see README "Notes").
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
# The original DAPO recipe used enforce_eager due to vLLM instability at the time.
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"

MEGATRON_TP="${MEGATRON_TP:-4}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_EP="${MEGATRON_EP:-8}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MICRO_FORWARD_BATCH_SIZE_PER_GPU="${MICRO_FORWARD_BATCH_SIZE_PER_GPU:-4}"
if [[ "$FULL_FT" == "1" ]]; then
  MICRO_TRAIN_BATCH_SIZE_PER_GPU="${MICRO_TRAIN_BATCH_SIZE_PER_GPU:-2}"
else
  MICRO_TRAIN_BATCH_SIZE_PER_GPU="${MICRO_TRAIN_BATCH_SIZE_PER_GPU:-4}"
fi
# Keep DAPO_MICRO_TRAIN_BATCH_SIZE in the client equal to MICRO_TRAIN_BATCH_SIZE_PER_GPU.

# DAPO loss settings (client sends eps_clip_low/high per request)
CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
LOSS_REDUCTION="${LOSS_REDUCTION:-token_mean_legacy}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# Off-policy correction: geometric sequence masking (see docs/content/docs/algorithms/off_policy_correction.mdx)
GEO_MASK_LOW="${GEO_MASK_LOW:-0.99}"
GEO_MASK_HIGH="${GEO_MASK_HIGH:-1.01}"

# LoRA alpha (rank is chosen by the client via --lora-rank). The Tinker SDK cannot send alpha, so it is
# set here; the reference LoRA run uses alpha = rank = 128.
LORA_ALPHA="${LORA_ALPHA:-128}"

# Ship standalone LoRA adapters to vLLM instead of merging them into the base weights.
# Required on the Tinker path: sampling addresses the policy by the Tinker model_id, and that name only
# exists on the engines when the adapter is registered via load_lora_adapter (merge_lora=true syncs
# merged weights under the base model name instead, so every sample request 404s). Ignored for full FT.
MERGE_LORA="${MERGE_LORA:-false}"

LORA_CONFIG=""
if [[ "$FULL_FT" != "1" ]]; then
  LORA_CONFIG=", \"trainer.policy.model.lora.alpha\": $LORA_ALPHA"
fi

DEFAULT_BACKEND_CONFIG=$(cat <<JSON
{
  "trainer.strategy": "megatron",
  "trainer.placement.colocate_all": true,
  "trainer.placement.policy_num_nodes": $NUM_NODES,
  "trainer.placement.policy_num_gpus_per_node": $NUM_GPUS_PER_NODE,
  "trainer.policy.megatron_config.tensor_model_parallel_size": $MEGATRON_TP,
  "trainer.policy.megatron_config.pipeline_model_parallel_size": $MEGATRON_PP,
  "trainer.policy.megatron_config.context_parallel_size": $MEGATRON_CP,
  "trainer.policy.megatron_config.expert_model_parallel_size": $MEGATRON_EP,
  "trainer.policy.megatron_config.expert_tensor_parallel_size": $MEGATRON_ETP,
  "trainer.policy.megatron_config.lora_config.merge_lora": $MERGE_LORA,
  "trainer.micro_forward_batch_size_per_gpu": $MICRO_FORWARD_BATCH_SIZE_PER_GPU,
  "trainer.micro_train_batch_size_per_gpu": $MICRO_TRAIN_BATCH_SIZE_PER_GPU,
  "trainer.max_prompt_length": $MAX_PROMPT_LENGTH,
  "generator.sampling_params.max_generate_length": $MAX_RESPONSE_LENGTH,
  "generator.eval_sampling_params.max_generate_length": $MAX_RESPONSE_LENGTH,
  "trainer.algorithm.policy_loss_type": "dual_clip",
  "trainer.algorithm.clip_ratio_c": $CLIP_RATIO_C,
  "trainer.algorithm.loss_reduction": "$LOSS_REDUCTION",
  "trainer.algorithm.use_kl_loss": false,
  "trainer.algorithm.off_policy_correction.sequence_mask_metric": "geometric",
  "trainer.algorithm.off_policy_correction.geo_mask_low": $GEO_MASK_LOW,
  "trainer.algorithm.off_policy_correction.geo_mask_high": $GEO_MASK_HIGH,
  "trainer.policy.optimizer_config.weight_decay": $WEIGHT_DECAY,
  "trainer.policy.optimizer_config.max_grad_norm": $MAX_GRAD_NORM,
  "generator.inference_engine.num_engines": $NUM_INFERENCE_ENGINES,
  "generator.inference_engine.tensor_parallel_size": $INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE,
  "generator.inference_engine.backend": "vllm",
  "generator.inference_engine.run_engines_locally": true,
  "generator.inference_engine.weight_sync_backend": "nccl",
  "generator.inference_engine.gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
  "generator.inference_engine.enforce_eager": $ENFORCE_EAGER,
  "generator.batched": true$LORA_CONFIG
}
JSON
)
BACKEND_CONFIG="${BACKEND_CONFIG:-$DEFAULT_BACKEND_CONFIG}"

uv run --isolated --extra tinker --extra megatron -m skyrl.tinker.api \
  --base-model "$BASE_MODEL" \
  --backend megatron \
  --port "$PORT" \
  --backend-config "$BACKEND_CONFIG" \
  "$@"
