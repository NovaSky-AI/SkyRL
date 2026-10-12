#!/usr/bin/env bash
# Start a Tinker API server for the LoRA-flags demo. One GPU is enough: the demo
# only trains (forward_backward + optim_step), so no inference engine is created.

set -euo pipefail

DEFAULT_BACKEND_CONFIG='{"trainer.placement.colocate_all": false, "trainer.placement.policy_num_gpus_per_node": 1, "trainer.placement.policy_num_nodes": 1}'
BACKEND_CONFIG="${BACKEND_CONFIG:-$DEFAULT_BACKEND_CONFIG}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"

uv run --extra tinker --extra fsdp -m skyrl.tinker.api \
  --base-model "$BASE_MODEL" \
  --backend fsdp \
  --port "$PORT" \
  --backend-config "$BACKEND_CONFIG" \
  "$@"
