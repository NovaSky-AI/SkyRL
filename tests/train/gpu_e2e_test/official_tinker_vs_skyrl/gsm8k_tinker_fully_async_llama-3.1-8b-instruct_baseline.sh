#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NOT PART OF CI.
#
# One-off comparison: SkyRL's local tinker server vs. the hosted Tinker API,
# both training meta-llama/Llama-3.1-8B-Instruct on GSM8K under the fully-async
# layout (trainer and inference on disjoint GPUs, max_steps_off_policy=4).
# Not referenced from any workflow or ci/*.yaml.
#
# Hyperparameters follow the tinker-cookbook math_rl README recipe for gsm8k
# (group_size=64, groups_per_batch=32, learning_rate=8e-5, max_tokens=1024),
# scaled to a short CI-style budget of max_steps=10.
# -----------------------------------------------------------------------------
set -euo pipefail

RUN_NAME="${RUN_NAME:-skyrl_baseline_Llama-3.1-8B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-gsm8k_tinker_fully_async_ci}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
SCRIPT_DIR=$(dirname $(realpath $0))
SKYRL_REPO_ROOT=$(realpath "$SCRIPT_DIR/../../..")
LOG_DIR="$HOME/tinker_fully_async_logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

REWARD_MIN_VALUE=0.0

# Non-colocated layout: 2 GPUs for the trainer (FSDP policy) and 2 GPUs for the
# inference engines (vLLM). micro_batch=4 (vs. CI default 8) to leave headroom
# for the larger 1024-token sequences and 8B params on a 2-way FSDP shard.
BACKEND_CONFIG='{"trainer.placement.colocate_all": false, "trainer.placement.policy_num_gpus_per_node": 2, "trainer.micro_forward_batch_size_per_gpu": 4, "trainer.micro_train_batch_size_per_gpu": 4, "generator.inference_engine.num_engines": 2, "generator.inference_engine.tensor_parallel_size": 1, "generator.inference_engine.backend": "vllm", "generator.inference_engine.run_engines_locally": true, "generator.inference_engine.weight_sync_backend": "nccl", "generator.inference_engine.async_engine": true, "generator.inference_engine.gpu_memory_utilization": 0.8, "generator.batched": true}'

setsid uv run --extra tinker --extra fsdp -m skyrl.tinker.api \
  --base-model "$MODEL_NAME" --backend fsdp --port 8000 \
  --backend-config "$BACKEND_CONFIG" >"$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM -- -$SERVER_PID 2>/dev/null || true; sleep 5; kill -KILL -- -$SERVER_PID 2>/dev/null || true' EXIT

deadline=$(( $(date +%s) + 1800 ))
until curl -sSf http://localhost:8000/docs >/dev/null 2>&1; do
  if (( $(date +%s) > deadline )); then
    echo "Tinker server did not become ready within 30 minutes" >&2
    tail -n 200 "$LOG_DIR/server.log" >&2 || true
    exit 1
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Tinker server exited early" >&2
    tail -n 200 "$LOG_DIR/server.log" >&2 || true
    exit 1
  fi
  sleep 5
done

COOKBOOK_DIR="$HOME/tinker-cookbook"
[ -d "$COOKBOOK_DIR" ] || git clone --depth 1 https://github.com/thinking-machines-lab/tinker-cookbook.git "$COOKBOOK_DIR"

cd "$COOKBOOK_DIR"
TINKER_API_KEY=tml-dummy uv run --extra math-rl --extra wandb --with tinker --with datasets --with torch \
  python -m tinker_cookbook.recipes.math_rl.train \
  base_url=http://localhost:8000 \
  model_name="$MODEL_NAME" \
  env=gsm8k \
  log_path="$LOG_DIR" \
  groups_per_batch=32 \
  group_size=64 \
  learning_rate=8e-5 \
  max_tokens=1024 \
  max_steps=10 \
  max_steps_off_policy=4 \
  eval_every=10000 \
  save_every=10000 \
  wandb_project="$PROJECT_NAME" \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists=delete

cd "$SKYRL_REPO_ROOT"
uv run --isolated --extra fsdp "$SCRIPT_DIR/get_summary.py" \
  --run_name "$RUN_NAME" --project_name "$PROJECT_NAME" \
  --asserts "env/all/reward/total >= $REWARD_MIN_VALUE"
