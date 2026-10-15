#!/usr/bin/env bash
# Sweep the three LoRA group configurations and write a comparison table.
#
# FSDP serves exactly one adapter per server lifetime (multi-tenant LoRA is
# Megatron-only), so each configuration gets its own server start.
#
#   bash examples/tinker/lora_flags/run_demo.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
STEPS="${STEPS:-30}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
OUT="${OUT:-lora_flags_results.json}"
LOG_DIR="${LOG_DIR:-/tmp/lora_flags_demo}"

mkdir -p "$LOG_DIR"
rm -f "$OUT"

port_is_free() {
  ! curl -sf "http://0.0.0.0:${1}/api/v1/healthz" > /dev/null 2>&1
}

wait_for_server() {
  local label="$1" port="$2"
  for _ in $(seq 1 180); do
    if curl -sf "http://0.0.0.0:${port}/api/v1/healthz" > /dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "server did not come up; see ${LOG_DIR}/server_${label}.log" >&2
  tail -40 "${LOG_DIR}/server_${label}.log" >&2 || true
  return 1
}

run_config() {
  local label="$1" attn_flag="$2" mlp_flag="$3" port="$4"
  echo "=============== ${label} (port ${port}) ==============="

  # Each configuration needs its own server: FSDP serves one adapter per server
  # lifetime, and a second create_model against a live server is rejected by the
  # LoRA signature check. A distinct port per configuration makes that explicit,
  # so a leaked server can never be mistaken for a fresh one.
  if ! port_is_free "$port"; then
    echo "port ${port} is already serving; refusing to reuse another run's server" >&2
    return 1
  fi

  # setsid puts the server in its own process group. Killing the bash wrapper
  # alone would leave the `uv run ... -m skyrl.tinker.api` child holding the port
  # and the GPU, so kill the whole group below.
  PORT="$port" BASE_MODEL="$BASE_MODEL" \
    setsid bash "${HERE}/run_tinker_server.sh" > "${LOG_DIR}/server_${label}.log" 2>&1 &
  local server_pid=$!
  # shellcheck disable=SC2064
  trap "kill -- -${server_pid} 2>/dev/null || true" RETURN

  wait_for_server "$label" "$port"
  uv run --extra tinker --extra fsdp python "${HERE}/lora_flags_demo.py" \
    --base-url "http://0.0.0.0:${port}/" \
    --base-model "$BASE_MODEL" \
    --label "$label" \
    --steps "$STEPS" \
    --out "$OUT" \
    "$attn_flag" "$mlp_flag"

  kill -- "-${server_pid}" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  # Wait for the port to actually close, so the next config cannot connect to a
  # dying server and so its GPU memory is released before the next one loads.
  for _ in $(seq 1 60); do
    port_is_free "$port" && break
    sleep 2
  done
}

run_config both      --train-attn    --train-mlp    "$PORT"
run_config attn_only --train-attn    --no-train-mlp "$((PORT + 1))"
run_config mlp_only  --no-train-attn --train-mlp    "$((PORT + 2))"

echo
echo "=============== summary ==============="
uv run --extra tinker --extra fsdp python - "$OUT" <<'PY'
import json, sys

with open(sys.argv[1]) as f:
    results = {r["label"]: r for r in json.load(f)}

# Report where the configurations actually separate. First/final loss are the two
# least informative points on this task: step 0 is identical by construction (LoRA
# starts at B=0, so the adapter is inert whatever it covers) and the final loss is
# 0 for every configuration once the fixed batch is memorised. The discriminating
# signal is how fast each one gets there.
THRESHOLD = 0.1

print(
    f"{'config':<12}{'attn':>6}{'mlp':>6}{'step 0':>9}{'step 3':>9}{'step 5':>9}"
    f"{'step 10':>9}{'steps<' + str(THRESHOLD):>11}{'mean':>9}"
)
for label in ("both", "attn_only", "mlp_only"):
    r = results.get(label)
    if r is None:
        continue
    losses = r["losses"]

    def at(i):
        return losses[i] if i < len(losses) else float("nan")

    converged = next((i for i, v in enumerate(losses) if v < THRESHOLD), None)
    print(
        f"{label:<12}{str(r['train_attn']):>6}{str(r['train_mlp']):>6}"
        f"{at(0):>9.4f}{at(3):>9.4f}{at(5):>9.4f}{at(10):>9.4f}"
        f"{(str(converged) if converged is not None else '-'):>11}"
        f"{sum(losses) / len(losses):>9.4f}"
    )

curves = {k: tuple(round(x, 6) for x in v["losses"]) for k, v in results.items()}
step0 = {round(v["losses"][0], 6) for v in results.values()}
print()
if len(step0) != 1:
    print(f"WARNING: step-0 loss differs across configurations ({step0}); the runs are not comparable.")
if len(set(curves.values())) == 1:
    print("WARNING: every configuration produced an identical curve -- the flags are being ignored.")
else:
    print(f"{len(set(curves.values()))} of {len(curves)} curves are distinct, so the flags reached PEFT.")
    print("Adapting both groups should converge fastest; it has the most adapted modules.")
PY
