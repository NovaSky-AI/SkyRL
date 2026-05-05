#!/usr/bin/env bash
# Launch all taste judge ablation runs in parallel.
#
# Each script runs in a background subshell with output captured to
# logs/taste-TIMESTAMP/<name>.log. After 15s a status table is printed;
# any script that has already exited is treated as an immediate failure
# and its last 10 log lines are shown.
#
# Required env vars (union of all scripts):
#   FLEET_API_KEY, WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#   ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF_TOKEN
#
# Usage:
#   export FLEET_API_KEY=... WANDB_API_KEY=... AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=... ANTHROPIC_API_KEY=... OPENROUTER_API_KEY=... HF_TOKEN=...
#   bash scripts/fleet-taste-launch-all.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="logs/taste-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

# --------------------------------------------------------------------------
# Script list: "name:script-path"
# --------------------------------------------------------------------------
ENTRIES=(
  "baseline:fleet-taste-baseline-launch.sh"
  "judge:fleet-taste-judge-launch.sh"
  "haiku-vision:fleet-taste-haiku-vision-launch.sh"
  "abl-actions:fleet-taste-ablation-actions-launch.sh"
  "abl-screenshots:fleet-taste-ablation-screenshots-launch.sh"
  "abl-both:fleet-taste-ablation-both-launch.sh"
  "abl-reasoning:fleet-taste-ablation-reasoning-launch.sh"
  "rel-haiku:fleet-taste-relative-haiku-launch.sh"
  "rel-sonnet:fleet-taste-relative-sonnet-launch.sh"
)

NAMES=()
PIDS=()

# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------
echo "Logs → $LOG_DIR/"
echo ""
for entry in "${ENTRIES[@]}"; do
  name="${entry%%:*}"
  script="${SCRIPT_DIR}/${entry##*:}"
  log="$LOG_DIR/${name}.log"

  bash "$script" > "$log" 2>&1 &
  pid=$!
  NAMES+=("$name")
  PIDS+=("$pid")
  printf "  %-22s PID=%-8s log=%s\n" "$name" "$pid" "$log"
done

# --------------------------------------------------------------------------
# Wait briefly, then report immediate failures
# --------------------------------------------------------------------------
echo ""
echo "Waiting 15s for immediate failures (env-var checks, auth errors)..."
sleep 15
echo ""

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
RST='\033[0m'

printf "%-22s  %s\n" "RUN" "STATUS"
printf "%-22s  %s\n" "---" "------"
any_failed=0
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  pid="${PIDS[$i]}"
  log="$LOG_DIR/${name}.log"

  if kill -0 "$pid" 2>/dev/null; then
    printf "%-22s  ${GRN}RUNNING${RST} (PID=%s)\n" "$name" "$pid"
  else
    wait "$pid" 2>/dev/null; code=$?
    if [ "$code" -eq 0 ]; then
      # Finished cleanly already (unlikely at 15s but handle it)
      printf "%-22s  ${GRN}DONE${RST} (exit 0)\n" "$name"
    else
      printf "%-22s  ${RED}FAILED${RST} (exit %s)\n" "$name" "$code"
      echo "    ↳ last 10 lines of $log:"
      tail -10 "$log" 2>/dev/null | sed 's/^/        /'
      any_failed=1
    fi
  fi
done

echo ""
echo "PIDs saved — to check status later:"
for i in "${!NAMES[@]}"; do
  printf "  %-22s  PID=%s  kill -0 %s\n" "${NAMES[$i]}" "${PIDS[$i]}" "${PIDS[$i]}"
done

echo ""
if [ "$any_failed" -eq 1 ]; then
  echo "One or more runs failed immediately. Check the logs above."
  exit 1
else
  echo "All runs appear healthy. Logs in $LOG_DIR/"
fi
