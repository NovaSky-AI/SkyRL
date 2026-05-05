#!/usr/bin/env bash
# Launch taste-trial: baseline run with taste disabled (floor=1.0).
#
# Usage:
#   export FLEET_API_KEY=... WANDB_API_KEY=... AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=... OPENROUTER_API_KEY=... HF_TOKEN=...
#   bash scripts/fleet-taste-baseline-launch.sh
set -eo pipefail

YAML="tasks/openenv-fleet-grpo-vl-taste.yaml"

: "${FLEET_API_KEY:?set FLEET_API_KEY}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY}"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
: "${HF_TOKEN:?set HF_TOKEN}"

CLUSTER="taste-trial"
RUN_NAME="fleet_qwen35_browser_use_baseline"

sky launch -c "$CLUSTER" --retry-until-up -y "$YAML" \
  --env RUN_ID="baseline" \
  --env FLEET_API_KEY="$FLEET_API_KEY" \
  --env WANDB_API_KEY="$WANDB_API_KEY" \
  --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --env OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  --env SKYRL_TASTE_DISABLED=1 \
  --env TASTE_FLOOR=1.0 \
  --env HF_TOKEN="$HF_TOKEN" &

SKY_PID=$!

# Start WandB health monitor in background. Waits up to 30m for the run to
# register, then watches for GHOST (no reward metrics after 60m) and STALL
# (no step advance for 90m). Kills the cluster on fault.
WANDB_API_KEY="$WANDB_API_KEY" python3 scripts/wandb_run_monitor.py \
  --project fleet-browser-use-grpo \
  --run-name "$RUN_NAME" \
  --sky-cluster "$CLUSTER" \
  --grace-minutes 60 \
  --stall-minutes 90 \
  --poll-interval 120 \
  > "wandb_monitor_${CLUSTER}.log" 2>&1 &

MONITOR_PID=$!
echo "[launch] Monitor PID=$MONITOR_PID  Sky PID=$SKY_PID"

wait $SKY_PID
SKY_EXIT=$?

# Training finished (or sky launch failed) — stop the monitor.
kill "$MONITOR_PID" 2>/dev/null || true
exit $SKY_EXIT
