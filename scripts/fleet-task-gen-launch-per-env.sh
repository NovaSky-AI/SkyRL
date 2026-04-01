#!/usr/bin/env bash
# Launch per-env task-gen experiments — one SkyPilot cluster per environment.
# Targets ~40 training steps per env by computing NUM_EPOCHS from seed counts.
#
# Usage:
#   export FLEET_API_KEY=... WANDB_API_KEY=... OPENROUTER_API_KEY=...
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
#   bash scripts/fleet-task-gen-launch-per-env.sh [env1 env2 ...]
#
# If no envs specified, defaults to: ticketmaster zillow outlook
set -euo pipefail

YAML="tasks/task-gen-grpo-qwen3_5-9b.yaml"
EVAL_RATIO="0.05"
TARGET_STEPS=40
BATCH_SIZE=12

# Required env vars
: "${FLEET_API_KEY:?set FLEET_API_KEY}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY}"

# Seed counts per env from v55 dataset (after EVAL_RATIO=0.05 split)
declare -A SEEDS=(
  [booking]=539 [budget]=567 [carlisle]=336 [outlook]=181
  [reddit]=505 [rops-mail]=44 [ticketmaster]=212 [zillow]=106
)

# Default envs if none specified on command line
ENVS=("${@:-ticketmaster zillow outlook}")
if [[ $# -eq 0 ]]; then
  ENVS=(ticketmaster zillow outlook)
fi

for env in "${ENVS[@]}"; do
  seeds=${SEEDS[$env]:-100}
  steps_per_epoch=$(( (seeds + BATCH_SIZE - 1) / BATCH_SIZE ))
  num_epochs=$(( (TARGET_STEPS + steps_per_epoch - 1) / steps_per_epoch ))
  total_steps=$(( steps_per_epoch * num_epochs ))

  echo "=== Launching task-gen-${env}: ${seeds} seeds, ${steps_per_epoch} steps/epoch, ${num_epochs} epochs (${total_steps} steps) ==="
  sky launch -c "task-gen-${env}" "$YAML" \
    --env TASK_GEN_ENV_CLASSES="$env" \
    --env NUM_EPOCHS="$num_epochs" \
    --env FLEET_API_KEY="$FLEET_API_KEY" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
    --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    --yes --async
done

echo ""
echo "Launched ${#ENVS[@]} clusters. Monitor with: sky status"
