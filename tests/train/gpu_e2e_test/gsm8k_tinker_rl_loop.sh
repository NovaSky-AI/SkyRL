#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="run_$(date +%Y%m%d%H)"
SCRIPT_DIR=$(dirname $(realpath $0))
LOG_DIR="$HOME/tinker_logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

# TODO: tighten thresholds after 3-5 nightly runs (5% allowance from min observed),
# matching the convention in gsm8k_colocate.sh.
REWARD_MIN_VALUE=0.0

BACKEND_CONFIG='{"trainer.placement.colocate_all": false, "trainer.placement.policy_num_gpus_per_node": 2, "generator.inference_engine.num_engines": 2, "generator.inference_engine.tensor_parallel_size": 1, "generator.inference_engine.backend": "vllm", "generator.inference_engine.run_engines_locally": true, "generator.inference_engine.weight_sync_backend": "nccl", "generator.inference_engine.async_engine": true, "generator.inference_engine.gpu_memory_utilization": 0.8, "generator.batched": true}'

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

cd "$COOKBOOK_DIR"
TINKER_API_KEY=tml-dummy uv run --extra math-rl --with tinker --with datasets --with torch \
  python -m tinker_cookbook.recipes.rl_loop \
  base_url=http://localhost:8000 \
  model_name="Qwen/Qwen3-0.6B" \
  log_path="$LOG_DIR" \
  batch_size=512 \
  group_size=8 \
  max_tokens=512 \
  save_every=10000

uv run --no-project python "$SCRIPT_DIR/check_tinker_metrics.py" \
  --metrics-file "$LOG_DIR/metrics.jsonl" \
  --asserts "reward/total >= $REWARD_MIN_VALUE"
