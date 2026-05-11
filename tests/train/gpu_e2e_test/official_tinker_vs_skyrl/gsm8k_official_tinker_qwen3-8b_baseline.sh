#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NOT PART OF CI.
#
# One-off comparison script: SkyRL's local tinker server vs. the hosted Tinker
# API, both training Qwen3-8B on GSM8K under the colocated layout. Used to
# generate side-by-side baselines in wandb; not referenced from any workflow
# or ci/*.yaml. Hyperparameters follow the tinker-cookbook math_rl README
# recommendation for Qwen3-8B (group_size=16, groups_per_batch=64,
# learning_rate=2e-5). max_steps stays low because this is a smoke-style
# convergence check, not a full training run.
# -----------------------------------------------------------------------------
set -euo pipefail

RUN_NAME="${RUN_NAME:-skyrl_baseline_Qwen3-8B}"
PROJECT_NAME="${PROJECT_NAME:-gsm8k_tinker_ci}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
SCRIPT_DIR=$(dirname $(realpath $0))
SKYRL_REPO_ROOT=$(realpath "$SCRIPT_DIR/../../..")
LOG_DIR="$HOME/tinker_logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

REWARD_MIN_VALUE=0.0

BACKEND_CONFIG='{"trainer.placement.colocate_all": true, "trainer.placement.policy_num_gpus_per_node": 4, "trainer.micro_forward_batch_size_per_gpu": 4, "trainer.micro_train_batch_size_per_gpu": 4, "generator.inference_engine.num_engines": 4, "generator.inference_engine.tensor_parallel_size": 1, "generator.inference_engine.backend": "vllm", "generator.inference_engine.run_engines_locally": true, "generator.inference_engine.weight_sync_backend": "nccl", "generator.inference_engine.async_engine": true, "generator.inference_engine.gpu_memory_utilization": 0.8, "generator.batched": true}'

# Start tinker server in its own process group so we can clean up the engine subprocess too.
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
  groups_per_batch=64 \
  group_size=16 \
  learning_rate=2e-5 \
  max_tokens=512 \
  max_steps=14 \
  eval_every=10000 \
  save_every=10000 \
  wandb_project="$PROJECT_NAME" \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists=delete

cd "$SKYRL_REPO_ROOT"
uv run --isolated --extra fsdp "$SCRIPT_DIR/get_summary.py" \
  --run_name "$RUN_NAME" --project_name "$PROJECT_NAME" \
  --asserts "env/all/reward/total >= $REWARD_MIN_VALUE"
