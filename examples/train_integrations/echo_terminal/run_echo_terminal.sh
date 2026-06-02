#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-examples/train_integrations/echo_terminal/configs/qwen3_8b_rl.yaml}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python -m examples.train_integrations.echo_terminal.entrypoint --config "$CONFIG_PATH"
