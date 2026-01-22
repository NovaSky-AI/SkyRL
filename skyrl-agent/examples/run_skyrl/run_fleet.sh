#!/bin/bash
# Fleet Task Training via SkyRL-Agent
#
# Usage:
#   FLEET_API_KEY=<key> WANDB_API_KEY=<key> ./run_fleet.sh
#
# Environment Variables:
#   FLEET_API_KEY     - Required: Fleet API key
#   WANDB_API_KEY     - Required: Weights & Biases API key
#   TASKS_FILE        - Path to tasks JSON file (default: data/fleet_booking_sample.json)
#   ENV_KEY           - Filter tasks by env_key (default: all)
#   MAX_TASKS         - Limit number of tasks (default: all)
#   MODALITY          - Task modality: tool_use or computer_use (default: tool_use)
#   MODEL_PATH        - Model to train (default: Qwen/Qwen2.5-1.5B-Instruct)
#   NUM_GPUS          - Number of GPUs (default: 1)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Defaults
TASKS_FILE="${TASKS_FILE:-$AGENT_DIR/data/fleet_booking_sample.json}"
MODALITY="${MODALITY:-tool_use}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"

# Validate required env vars
if [ -z "${FLEET_API_KEY:-}" ]; then
    echo "ERROR: FLEET_API_KEY is required"
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is required"
    exit 1
fi

echo "=== Fleet Task Training ==="
echo "Tasks file: $TASKS_FILE"
echo "Modality: $MODALITY"
echo "Model: $MODEL_PATH"
echo "GPUs: $NUM_GPUS"
echo "Max iterations: $MAX_ITERATIONS"
echo ""

# Change to agent directory
cd "$AGENT_DIR"

# Setup virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.12 --seed
fi

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv sync
uv pip install wandb
uv pip install "git+https://github.com/fleet-ai/OpenEnv.git@deniz/fleet_client" fleet-python

# Login to WandB
python3 -c "import wandb; wandb.login(relogin=True, key='$WANDB_API_KEY')"

# Prepare dataset
echo "Preparing dataset..."
DATA_DIR="$HOME/data/fleet/${MODALITY}"
mkdir -p "$DATA_DIR"

python -c "
import json
import pandas as pd
from pathlib import Path

tasks_file = '$TASKS_FILE'
output_dir = Path('$DATA_DIR')

with open(tasks_file) as f:
    tasks = json.load(f)

if isinstance(tasks, dict) and 'tasks' in tasks:
    tasks = tasks['tasks']

# Convert to dataframe format expected by skyrl-agent
records = []
for task in tasks:
    records.append({
        'instance': task,
        'data_source': 'fleet',
    })

df = pd.DataFrame(records)

# Split 90/10
train_size = int(len(df) * 0.9)
train_df = df[:train_size]
val_df = df[train_size:]

train_df.to_parquet(output_dir / 'train.parquet')
val_df.to_parquet(output_dir / 'validation.parquet')

print(f'Created {len(train_df)} train and {len(val_df)} validation samples')
"

# Start Ray
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
if ! ray status --address 127.0.0.1:6479 >/dev/null 2>&1; then
    ray start --head --disable-usage-stats --port 6479
fi

# Wait for Ray
for i in $(seq 1 24); do
    if ray status --address 127.0.0.1:6479 >/dev/null 2>&1; then
        break
    fi
    sleep 5
done

# Run training
echo "Starting training..."
export TASKS_FILE
export FLEET_API_KEY
export ENV_KEY="${ENV_KEY:-}"
export MAX_TASKS="${MAX_TASKS:-}"

python -m skyrl_agent.auto \
    --config examples/run_skyrl/skyrl_fleet.yaml \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/validation.parquet']" \
    generator.max_iterations=$MAX_ITERATIONS \
    trainer.epochs=$NUM_EPOCHS \
    trainer.policy.model.path="$MODEL_PATH" \
    trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
    trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
    generator.num_inference_engines=$NUM_GPUS \
    trainer.logger=wandb \
    trainer.project_name=fleet-task-training \
    trainer.run_name="fleet_${MODALITY}_$(date +%Y%m%d_%H%M%S)"
