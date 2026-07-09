#!/usr/bin/env bash

# Starts a SkyRL Tinker API server with the Megatron backend and Rollout Routing
# Replay (R3) enabled for an MoE model (Qwen3.6-35B-A3B) on 1 node of 8 GPUs.
#
# `trainer.policy.megatron_config.moe_enable_routing_replay=true` turns on R3;
# SkyRL auto-enables routed-experts capture in the inference engine to match.
#
# Usage:
#   bash examples/tinker/router_replay/run_tinker_server_megatron.sh
#
#   # Then, in another terminal:
#   TINKER_API_KEY=tml-dummy uv run --extra tinker --with torch --with requests \
#       python examples/tinker/router_replay/router_replay_demo.py

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
PORT="${PORT:-8000}"

DEFAULT_BACKEND_CONFIG='{
  "trainer.strategy": "megatron",
  "trainer.placement.colocate_all": true,
  "trainer.placement.policy_num_nodes": 1,
  "trainer.placement.policy_num_gpus_per_node": 8,
  "trainer.policy.megatron_config.tensor_model_parallel_size": 2,
  "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
  "trainer.policy.megatron_config.context_parallel_size": 1,
  "trainer.policy.megatron_config.expert_model_parallel_size": 8,
  "trainer.policy.megatron_config.expert_tensor_parallel_size": 1,
  "trainer.policy.megatron_config.moe_enable_routing_replay": true,
  "trainer.policy.language_model_only": true,
  "trainer.micro_train_batch_size_per_gpu": 1,
  "trainer.micro_forward_batch_size_per_gpu": 1,
  "generator.inference_engine.num_engines": 1,
  "generator.inference_engine.tensor_parallel_size": 8,
  "generator.inference_engine.backend": "vllm",
  "generator.inference_engine.run_engines_locally": true,
  "generator.inference_engine.weight_sync_backend": "nccl",
  "generator.inference_engine.gpu_memory_utilization": 0.6,
  "generator.inference_engine.distributed_executor_backend": "mp",
  "generator.inference_engine.language_model_only": true,
  "generator.inference_engine.engine_init_kwargs": {"gdn_prefill_backend": "triton"}
}'
BACKEND_CONFIG="${BACKEND_CONFIG:-$DEFAULT_BACKEND_CONFIG}"

uv run --extra tinker --extra megatron -m skyrl.tinker.api \
  --base-model "$MODEL_NAME" \
  --backend megatron \
  --port "$PORT" \
  --backend-config "$BACKEND_CONFIG" \
  "$@"
