#!/usr/bin/env bash

set -euo pipefail

DEFAULT_BACKEND_CONFIG='{"trainer.placement.colocate_all": true, "trainer.placement.policy_num_gpus_per_node": 4, "trainer.placement.critic_num_gpus_per_node": 4, "trainer.critic.model.path": "Qwen/Qwen2.5-1.5B-Instruct", "trainer.micro_forward_batch_size_per_gpu": 64, "trainer.micro_train_batch_size_per_gpu": 64, "generator.inference_engine.num_engines": 4, "generator.inference_engine.tensor_parallel_size": 1, "generator.inference_engine.backend": "vllm", "generator.inference_engine.run_engines_locally": true, "generator.inference_engine.weight_sync_backend": "nccl", "generator.inference_engine.async_engine": true, "generator.inference_engine.gpu_memory_utilization": 0.8, "generator.batched": true}'
BACKEND_CONFIG="${BACKEND_CONFIG:-$DEFAULT_BACKEND_CONFIG}"

uv run --extra tinker --extra fsdp -m skyrl.tinker.api \
  --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
  --backend fsdp \
  --port 8000 \
  --backend-config "$BACKEND_CONFIG" \
  "$@"
