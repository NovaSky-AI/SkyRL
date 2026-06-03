#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="run_$(date +%Y%m%d%H)"
PROJECT_NAME="gsm8k_tinker_fully_async_ci"
SCRIPT_DIR=$(dirname $(realpath $0))
SKYRL_REPO_ROOT=$(realpath "$SCRIPT_DIR/../../..")
LOG_DIR="$HOME/tinker_fully_async_logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

# TODO: tighten thresholds after 3-5 nightly runs (5% allowance from min observed),
# matching the convention in gsm8k_colocate.sh.
REWARD_MIN_VALUE=0.0

# Non-colocated layout: 2 GPUs for the trainer (FSDP policy) and 2 GPUs for the
# inference engines (vLLM). colocate_all=false keeps training and inference on
# disjoint GPUs so they can run concurrently with the cookbook's async loop.
BACKEND_CONFIG='{"trainer.placement.colocate_all": false, "trainer.placement.policy_num_gpus_per_node": 2, "trainer.micro_forward_batch_size_per_gpu": 8, "trainer.micro_train_batch_size_per_gpu": 8, "generator.inference_engine.num_engines": 2, "generator.inference_engine.tensor_parallel_size": 1, "generator.inference_engine.backend": "vllm", "generator.inference_engine.run_engines_locally": true, "generator.inference_engine.weight_sync_backend": "nccl", "generator.inference_engine.async_engine": true, "generator.inference_engine.gpu_memory_utilization": 0.8, "generator.batched": true}'

# Start tinker server in its own process group so we can clean up the engine subprocess too.
setsid uv run --extra tinker --extra fsdp -m skyrl.tinker.api \
  --base-model "Qwen/Qwen3-0.6B" --backend fsdp --port 8000 \
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

# math_rl.train builds on tinker_cookbook/rl/train.py and exposes wandb_project /
# wandb_name natively, so we get the same wandb-driven flow as the other E2E
# nightlies (no client-side metrics publisher needed).
# max_steps_off_policy enables the cookbook's async mode: sampling and training
# run concurrently and samples up to N steps stale are accepted into the batch.
cd "$COOKBOOK_DIR"
TINKER_API_KEY=tml-dummy uv run --extra math-rl --extra wandb --with tinker --with datasets --with torch \
  python -m tinker_cookbook.recipes.math_rl.train \
  base_url=http://localhost:8000 \
  model_name="Qwen/Qwen3-0.6B" \
  env=gsm8k \
  log_path="$LOG_DIR" \
  groups_per_batch=512 \
  group_size=4 \
  max_tokens=512 \
  max_steps=14 \
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
