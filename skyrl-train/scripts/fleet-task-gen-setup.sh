#!/usr/bin/env bash
# Task-gen specific setup: calls common setup + prepares task-gen dataset
#
# Usage (from SkyPilot YAML setup block):
#   bash skyrl-train/scripts/fleet-task-gen-setup.sh
#
# Required env vars: FLEET_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
#   MODALITY, DATA_VERSION, S3_DATASET_BUCKET, SUPABASE_URL, SUPABASE_KEY
# Optional env vars: MAX_TASKS, ENV_KEYS, EVAL_RATIO
set -euo pipefail

# --- Common setup (venv, deps, Qwen3.5 extras, dataset download) ---
# --extra-pip supabase: task-gen needs Supabase client for DB schema discovery
# --skip-prepare: task-gen uses prepare_task_gen_dataset (not default prepare_dataset)
bash skyrl-train/scripts/fleet-common-setup.sh \
  --openenv-branch deniz/db-query-tools \
  --extra-setup skyrl-train/scripts/fleet-qwen35-extra-setup.sh \
  --extra-pip supabase \
  --skip-prepare

# --- Task-gen dataset preparation (parquet with schema/tools context per env) ---
cd skyrl-train
source .venv/bin/activate

# Auto-detect data root (same logic as fleet-common-setup.sh)
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
  DATA_ROOT="/workspace"
else
  DATA_ROOT="$HOME"
fi

TASKS_FILE="$DATA_ROOT/data/fleet/tasks_${MODALITY}.json"
DATA_DIR="$DATA_ROOT/data/fleet/task_gen"
TOOLS_CACHE="$DATA_ROOT/data/fleet/tools_cache.json"
SCHEMA_CACHE="$DATA_ROOT/data/fleet/schema_cache.json"

PREPARE_CMD="python -m integrations.fleet.prepare_task_gen_dataset --tasks-json $TASKS_FILE --output-dir $DATA_DIR --mode grpo --tools-cache $TOOLS_CACHE --schema-cache $SCHEMA_CACHE"
[ -n "${MAX_TASKS:-}" ] && PREPARE_CMD="$PREPARE_CMD --max-tasks $MAX_TASKS"
[ -n "${ENV_KEYS:-}" ] && PREPARE_CMD="$PREPARE_CMD --env-keys $ENV_KEYS"
[ -n "${EVAL_RATIO:-}" ] && PREPARE_CMD="$PREPARE_CMD --eval-ratio $EVAL_RATIO"
eval "$PREPARE_CMD"

echo "=== Task-Gen Setup Complete ==="
