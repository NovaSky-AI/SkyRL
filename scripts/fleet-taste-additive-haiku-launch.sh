#!/usr/bin/env bash
# Launch taste-additive-haiku: additive reward r = binary + α·taste (no gating).
#
# Judge: relative_haiku (same as relative-haiku baseline) so results are comparable.
# Reward: r_i = verifier_reward + TASTE_ALPHA * taste_score, scored for ALL rollouts
#         including fails. Expected +33% advantage variance at α=0.5 vs binary-only.
#
# See research/judge/variance_report.md §5 for derivation.
#
# Usage:
#   export FLEET_API_KEY=... WANDB_API_KEY=... AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=... ANTHROPIC_API_KEY=... HF_TOKEN=...
#   [TASTE_ALPHA=0.5] bash scripts/fleet-taste-additive-haiku-launch.sh
set -eo pipefail

YAML="tasks/openenv-fleet-grpo-vl-additive-haiku.yaml"

: "${FLEET_API_KEY:?set FLEET_API_KEY}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY}"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
: "${HF_TOKEN:?set HF_TOKEN}"

TASTE_ALPHA="${TASTE_ALPHA:-0.5}"
CLUSTER="taste-additive-haiku"
RUN_NAME="fleet_qwen35_browser_use_additive-haiku-alpha${TASTE_ALPHA}"

sky launch -c "$CLUSTER" --retry-until-up -y "$YAML" \
  --env RUN_ID="additive-haiku-judge" \
  --env TASTE_ALPHA="$TASTE_ALPHA" \
  --env FLEET_API_KEY="$FLEET_API_KEY" \
  --env WANDB_API_KEY="$WANDB_API_KEY" \
  --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
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
