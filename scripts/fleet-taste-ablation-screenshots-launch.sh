#!/usr/bin/env bash
# Ablation B — Screenshots only (no action trace).
# Haiku judge receives: TASK + up to 4 screenshots. ACTIONS suppressed.
# Tests whether visual state alone is sufficient — especially for ui_grounding.
# If this matches or beats actions-only, the action trace is not load-bearing.
#
# Usage:
#   export FLEET_API_KEY=... WANDB_API_KEY=... AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=... ANTHROPIC_API_KEY=... HF_TOKEN=...
#   bash scripts/fleet-taste-ablation-screenshots-launch.sh
set -eo pipefail

YAML="tasks/openenv-fleet-grpo-vl-taste.yaml"

: "${FLEET_API_KEY:?set FLEET_API_KEY}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY}"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
: "${HF_TOKEN:?set HF_TOKEN}"

CLUSTER="taste-ablation-screenshots"
RUN_NAME="fleet_qwen35_browser_use_ablation-screenshots"

sky launch -c "$CLUSTER" --retry-until-up -y "$YAML" \
  --env RUN_ID="ablation-screenshots" \
  --env FLEET_API_KEY="$FLEET_API_KEY" \
  --env WANDB_API_KEY="$WANDB_API_KEY" \
  --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  --env SKYRL_TASTE_PROVIDER=haiku_vision \
  --env SKYRL_TASTE_MODEL=claude-haiku-4-5-20251001 \
  --env SKYRL_TASTE_BLIND_OUTCOME=1 \
  --env SKYRL_TASTE_BLIND_ACTIONS=1 \
  --env SKYRL_TASTE_BLIND_SCREENSHOTS=0 \
  --env TASTE_FLOOR=0.1 \
  --env HF_TOKEN="$HF_TOKEN" &

SKY_PID=$!

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

kill "$MONITOR_PID" 2>/dev/null || true
exit $SKY_EXIT
