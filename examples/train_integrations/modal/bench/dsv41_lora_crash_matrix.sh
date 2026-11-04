#!/bin/bash
# Runs the DeepSeek-V4.1 async LoRA recipe once per variant on one Ray cluster. A variant stops at the first
# vLLM engine failure or after MAX_TRAINING_STEPS steps, and the next one starts.
#
# Each argument is "name|recipe env assignments|recipe overrides", e.g.
#   bash dsv41_lora_crash_matrix.sh "base||" "eager||generator.inference_engine.enforce_eager=true"
# Per variant: <OUT_DIR>/<name>.log (driver output) and a summary line in <OUT_DIR>/summary.txt.
set -uo pipefail

: "${OUT_DIR:=/root/data/skyrl-logs/dsv41_crash_matrix}"
: "${VARIANT_TIMEOUT_S:=3600}"
mkdir -p "$OUT_DIR"
CRASH_RE="illegal memory access|EngineDeadError|Training failed at step|Cannot connect to host"

for spec in "$@"; do
  IFS='|' read -r name env_assigns overrides <<< "$spec"
  log="$OUT_DIR/$name.log"
  start=$(date +%s)
  echo "=== variant $name: env [$env_assigns] overrides [$overrides]"
  # shellcheck disable=SC2086
  setsid env $env_assigns \
    RUN_NAME="dsv41_crash_matrix_$name" \
    bash examples/train/deepseek_v4_flash/run_async_multiturn_dsv41_flash_lora.sh \
      trainer.ckpt_interval=-1 trainer.log_path="$OUT_DIR/$name" $overrides > "$log" 2>&1 &
  pid=$!
  result=timeout
  while kill -0 "$pid" 2>/dev/null; do
    if grep -qE "$CRASH_RE" "$log"; then
      result=crash
      sleep 20  # lets the engine log its traceback
      break
    fi
    if (( $(date +%s) - start > VARIANT_TIMEOUT_S )); then
      break
    fi
    sleep 10
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null; sleep 15; kill -KILL -- "-$pid" 2>/dev/null
  elif [ "$result" = timeout ]; then
    if wait "$pid"; then result=clean; else result=failed; fi
    grep -qE "$CRASH_RE" "$log" && result=crash
  fi
  steps=$(grep -c "Finished: 'step'" "$log")
  line="$name result=$result steps=$steps seconds=$(( $(date +%s) - start ))"
  echo "$line" | tee -a "$OUT_DIR/summary.txt"
  grep -nE "$CRASH_RE" "$log" "$OUT_DIR/$name"/infra-*.log 2>/dev/null | head -5 | cut -c1-300
  # Gives Ray time to reap the killed job's actors before the next variant starts.
  sleep 30
done
cat "$OUT_DIR/summary.txt"
