#!/bin/bash
# Modal: long-horizon (~200k-token) DSv4.1-Flash LoRA RL (see e2e_long_args.sh). Env (MODEL_PATH, DATA_DIR, MAX_TURNS,
# TRAIN_BATCH_SIZE, N_SAMPLES_PER_PROMPT, MAX_TRAINING_STEPS, RUN_NAME, CKPT_PATH, OFFLOAD_GB, MAX_MODEL_LEN, OUT) from
# the launcher.
set -o pipefail
cd "$(dirname "$0")/../../../.."
mkdir -p "$OUT"
source examples/train_integrations/modal/bench/e2e_long_args.sh
bash examples/train/deepseek_v4_flash/run_async_multiturn_dsv41_flash_lora.sh "${LONG_ARGS[@]}" trainer.log_path="$OUT" \
  > "$OUT/run.log" 2>&1
rc=$?
echo "rc=$rc" >> "$OUT/run.log"
exit $rc
