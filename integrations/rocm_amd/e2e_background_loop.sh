#!/usr/bin/env bash
# Background E2E loop: verify + GRPO on MI355X until "Training done!" or ALL VERIFY passes with GRPO.
# Writes status every 10 minutes to reports/e2e_status.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${ROOT}/integrations/rocm_amd/reports"
STATUS_LOG="${REPORT_DIR}/e2e_status.log"
INTERVAL="${E2E_STATUS_INTERVAL_SEC:-600}"
MAX_ATTEMPTS="${E2E_MAX_ATTEMPTS:-20}"

mkdir -p "${REPORT_DIR}"

log_status() {
  local msg="$1"
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[${ts}] ${msg}" | tee -a "${STATUS_LOG}"
}

check_success() {
  local logfile="$1"
  [[ -f "${logfile}" ]] || return 1
  if grep -q "Training done!" "${logfile}" 2>/dev/null; then
    return 0
  fi
  if grep -q "ALL VERIFY CHECKS PASSED" "${logfile}" 2>/dev/null \
    && grep -q "RUN_GRPO_SMOKE=1" "${logfile}" 2>/dev/null \
    && grep -q "Starting Megatron GRPO" "${logfile}" 2>/dev/null; then
    grep -q "Training done!" "${logfile}" 2>/dev/null
    return $?
  fi
  return 1
}

summarize_failure() {
  local logfile="$1"
  [[ -f "${logfile}" ]] || { echo "no log file"; return; }
  grep -E "^(FAIL:|ERROR:|srun: error|Traceback|Training done|ALL VERIFY)" "${logfile}" 2>/dev/null | tail -5 \
    || tail -3 "${logfile}"
}

write_snapshot() {
  local attempt="$1" phase="$2" logfile="$3"
  {
    echo "--- snapshot attempt=${attempt} phase=${phase} ---"
    squeue -u "${USER}" 2>/dev/null || true
    sinfo -p rad -o "%N %T" 2>/dev/null | head -5 || true
    if [[ -n "${logfile}" && -f "${logfile}" ]]; then
      echo "log=${logfile} size=$(wc -c < "${logfile}") lines=$(wc -l < "${logfile}")"
      tail -8 "${logfile}" 2>/dev/null || true
    fi
    echo "---"
  } >> "${STATUS_LOG}"
}

log_status "E2E background loop started (interval=${INTERVAL}s max_attempts=${MAX_ATTEMPTS})"
log_status "Branch=$(cd "${ROOT}" && git rev-parse --short HEAD 2>/dev/null) main=$(cd "${ROOT}" && git rev-parse --short origin/main 2>/dev/null)"

attempt=0
if [ -n "${FORCE_VLLM_REBUILD:-}" ]; then
  force_rebuild="${FORCE_VLLM_REBUILD}"
elif find "${ROOT}/integrations/rocm_amd/.vllm_rocm_cache/wheels" \
  -mindepth 2 -name 'vllm-0.20*.whl' -print -quit 2>/dev/null | grep -q .; then
  force_rebuild=0
else
  force_rebuild=1
fi

while [[ "${attempt}" -lt "${MAX_ATTEMPTS}" ]]; do
  attempt=$((attempt + 1))
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  run_log="${REPORT_DIR}/cluster_e2e_bg_${ts}.log"

  log_status "=== Attempt ${attempt}/${MAX_ATTEMPTS} FORCE_VLLM_REBUILD=${force_rebuild} ==="

  # If another srun e2e is already blocking, wait for it instead of duplicating.
  if pgrep -f "run_on_cluster.sh e2e" >/dev/null 2>&1; then
    log_status "Existing e2e srun detected; monitoring until it finishes..."
    existing_log="$(ls -1t "${REPORT_DIR}"/cluster_e2e_*.log 2>/dev/null | head -1 || true)"
    run_log="${existing_log:-${run_log}}"
    while pgrep -f "run_on_cluster.sh e2e" >/dev/null 2>&1; do
      write_snapshot "${attempt}" "waiting-existing" "${run_log}"
      sleep "${INTERVAL}"
    done
  else
    log_status "Launching: SLURM_GPUS=2 FORCE_VLLM_REBUILD=${force_rebuild} run_on_cluster.sh e2e"
    cd "${ROOT}"
    SLURM_GPUS=2 SLURM_TIME=02:00:00 FORCE_VLLM_REBUILD="${force_rebuild}" \
      bash integrations/rocm_amd/run_on_cluster.sh e2e \
      > "${run_log}" 2>&1 &
    runner_pid=$!
    log_status "Runner pid=${runner_pid} log=${run_log}"

    elapsed=0
    while kill -0 "${runner_pid}" 2>/dev/null; do
      if (( elapsed > 0 && elapsed % INTERVAL == 0 )); then
        log_status "Attempt ${attempt} still running (${elapsed}s elapsed)"
        write_snapshot "${attempt}" "running" "${run_log}"
      fi
      sleep 60
      elapsed=$((elapsed + 60))
    done
    wait "${runner_pid}" || true
  fi

  if check_success "${run_log}"; then
    log_status "SUCCESS: E2E GRPO complete. log=${run_log}"
    exit 0
  fi

  log_status "Attempt ${attempt} FAILED. Summary:"
  summarize_failure "${run_log}" | while read -r line; do log_status "  ${line}"; done
  write_snapshot "${attempt}" "failed" "${run_log}"

  # After first successful vLLM build, stop forcing rebuild unless compat failed.
  if grep -q "built vllm" "${run_log}" 2>/dev/null \
    || grep -q "PASS: vLLM SkyRL API compatibility" "${run_log}" 2>/dev/null; then
    force_rebuild=0
    log_status "Next attempt: FORCE_VLLM_REBUILD=0 (reuse vLLM wheel)"
  fi
  if grep -q "atomicAdd.*ambiguous\|could not patch atomicAdd" "${run_log}" 2>/dev/null; then
    force_rebuild=1
    log_status "Next attempt: FORCE_VLLM_REBUILD=1 (GPTQ build issue)"
  fi

  log_status "Sleeping ${INTERVAL}s before retry..."
  sleep "${INTERVAL}"
done

log_status "GAVE UP after ${MAX_ATTEMPTS} attempts. See ${REPORT_DIR}/cluster_e2e_bg_*.log"
exit 1
