#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NOT PART OF CI.
#
# One-off comparison: SkyRL's local tinker server (colocated layout) vs. the
# hosted Tinker API, training meta-llama/Llama-3.1-8B-Instruct on GSM8K.
# Counterpart to gsm8k_tinker_fully_async_llama-3.1-8b-instruct_baseline.sh
# but with colocate_all=true so the trainer and inference engines share all
# 4 GPUs (4 vLLM engines, sequential sample/train). Not referenced from any
# workflow or ci/*.yaml.
#
# HPs follow the cookbook math_rl README gsm8k recipe (group_size=64,
# groups_per_batch=32, learning_rate=8e-5, max_tokens=1024) at a 10-step
# CI-style budget.
# -----------------------------------------------------------------------------
set -euo pipefail

RUN_NAME="${RUN_NAME:-skyrl_baseline_Llama-3.1-8B-Instruct_colocated}"
PROJECT_NAME="${PROJECT_NAME:-gsm8k_tinker_ci}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
SCRIPT_DIR=$(dirname $(realpath $0))
SKYRL_REPO_ROOT=$(realpath "$SCRIPT_DIR/../../..")
LOG_DIR="$HOME/tinker_logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

REWARD_MIN_VALUE=0.0

# Colocated: 4 GPUs shared between FSDP trainer and 4 vLLM engines (TP=1).
# micro_batch=4 to leave headroom for max_tokens=1024 with 8B params.
BACKEND_CONFIG='{"trainer.placement.colocate_all": true, "trainer.placement.policy_num_gpus_per_node": 4, "trainer.micro_forward_batch_size_per_gpu": 4, "trainer.micro_train_batch_size_per_gpu": 4, "generator.inference_engine.num_engines": 4, "generator.inference_engine.tensor_parallel_size": 1, "generator.inference_engine.backend": "vllm", "generator.inference_engine.run_engines_locally": true, "generator.inference_engine.weight_sync_backend": "nccl", "generator.inference_engine.async_engine": true, "generator.inference_engine.gpu_memory_utilization": 0.8, "generator.batched": true}'

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
  eval_every=10000 \
  save_every=10000 \
  wandb_project="$PROJECT_NAME" \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists=delete

cd "$SKYRL_REPO_ROOT"
uv run --isolated --extra fsdp "$SCRIPT_DIR/get_summary.py" \
  --run_name "$RUN_NAME" --project_name "$PROJECT_NAME" \
  --asserts "env/all/reward/total >= $REWARD_MIN_VALUE"
