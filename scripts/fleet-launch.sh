#!/usr/bin/env bash
# Preflight-gated wrapper around `sky launch`.
#
# Use this in place of `sky launch` for any SkyRL-Fleet job. The wrapper
# runs scripts/fleet-preflight.sh first; if every check passes it then
# invokes `sky launch` with the supplied arguments. If any check fails,
# `sky launch` is never called — so we don't burn provisioning time on a
# job that's guaranteed to die in setup.
#
# Argument convention:
#
#   bash scripts/fleet-launch.sh [sky-launch args...]
#       — no `--`: every token is forwarded to `sky launch`. Preflight runs
#         with default settings (FLEET / WANDB / AWS required, gcloud + sky
#         + liveness all enabled).
#
#   bash scripts/fleet-launch.sh [preflight args...] -- [sky-launch args...]
#       — with `--`: tokens before `--` are forwarded verbatim to
#         scripts/fleet-preflight.sh (e.g. `--require OPENROUTER_API_KEY`,
#         `--skip-gcloud`, `--skip-liveness`). Tokens after `--` go to
#         `sky launch`.
#
# Examples:
#
#   # Standard launch — uses default preflight settings.
#   bash scripts/fleet-launch.sh tasks/openenv-fleet-grpo-vl.yaml \
#     --env FLEET_API_KEY="$FLEET_API_KEY" \
#     --env WANDB_API_KEY="$WANDB_API_KEY" \
#     --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
#     --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
#
#   # Task-gen launch — also require OPENROUTER_API_KEY.
#   bash scripts/fleet-launch.sh --require OPENROUTER_API_KEY -- \
#     tasks/task-gen-grpo-qwen3_5-9b.yaml --env FLEET_API_KEY=... ...
#
#   # Non-GCP launch — skip the gcloud check.
#   bash scripts/fleet-launch.sh --skip-gcloud -- tasks/foo.yaml --env ...
#
# Environment:
#   SKIP_PREFLIGHT=1   Bypass the preflight entirely (escape hatch; avoid
#                      using this in scripts — it defeats the safety net).
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

PREFLIGHT_ARGS=()
SKY_ARGS=()
SAW_DASHDASH=false

# Split argv on a literal `--`. Without `--`, every token goes to sky.
for arg in "$@"; do
  if [ "$SAW_DASHDASH" = true ]; then
    SKY_ARGS+=("$arg")
  elif [ "$arg" = "--" ]; then
    SAW_DASHDASH=true
  else
    PREFLIGHT_ARGS+=("$arg")
  fi
done
if [ "$SAW_DASHDASH" = false ]; then
  SKY_ARGS=(${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"})
  PREFLIGHT_ARGS=()
fi

if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
  echo "WARNING: SKIP_PREFLIGHT=1 — bypassing fleet-preflight.sh" >&2
else
  # Empty-array safe expansion (works under `set -u`).
  bash scripts/fleet-preflight.sh ${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"}
fi

if [ ${#SKY_ARGS[@]} -eq 0 ]; then
  echo "ERROR: no arguments to forward to 'sky launch'" >&2
  echo "Usage: bash scripts/fleet-launch.sh [preflight args...] -- <yaml> [sky launch args...]" >&2
  echo "   or: bash scripts/fleet-launch.sh <yaml> [sky launch args...]" >&2
  exit 1
fi

if ! command -v sky >/dev/null 2>&1; then
  echo "ERROR: sky CLI not found on PATH" >&2
  exit 1
fi

echo "=== Preflight passed; running: sky launch ${SKY_ARGS[*]} ==="
exec sky launch "${SKY_ARGS[@]}"
