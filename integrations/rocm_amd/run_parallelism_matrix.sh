#!/usr/bin/env bash
# Sweep Megatron / vLLM parallelism layouts on AMD.
#
# Inside a ROCm container with the stack already installed:
#   bash integrations/rocm_amd/run_parallelism_matrix.sh
#
# Optional:
#   MATRIX_FILTER=tp2     # substring match on case names
#   MATRIX_GPUS=2         # default NUM_GPUS
#   SKIP_INSTALL=1
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

REPORT_DIR="${ROOT}/integrations/rocm_amd/reports"
RESULT_MD="${REPORT_DIR}/PARALLELISM_MATRIX.md"
mkdir -p "${REPORT_DIR}"

MATRIX_GPUS="${MATRIX_GPUS:-${NUM_GPUS:-2}}"
MATRIX_NODES="${MATRIX_NODES:-${NUM_NODES:-1}}"
FILTER="${MATRIX_FILTER:-}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
  echo "SKIP_INSTALL=1: not running install_full_stack.sh"
elif python3 -c "from megatron.bridge import AutoBridge; import vllm" >/dev/null 2>&1; then
  echo "Stack already importable; skipping install"
else
  bash integrations/rocm_amd/install_full_stack.sh
fi

python3 integrations/rocm_amd/probe_megatron_compat.py || true

declare -a CASES=()
# name|gpus|nodes|megatron_tp|megatron_pp|megatron_cp|megatron_ep|vllm_tp|num_engines
CASES+=(
  "sn2_megatron_dp2_tp1_pp1_vllm_tp2|2|1|1|1|1|1|2|1"
  "sn2_megatron_tp2_vllm_tp2|2|1|2|1|1|1|2|1"
  "sn2_megatron_pp2_vllm_tp2|2|1|1|2|1|1|2|1"
  "sn2_megatron_cp2_vllm_tp2|2|1|1|1|2|1|2|1"
  "sn2_vllm_dp2_engines_tp1|2|1|1|1|1|1|1|2"
)

if [ "${MATRIX_GPUS}" -ge 4 ]; then
  CASES+=(
    "sn4_megatron_tp2_dp2_vllm_tp4|4|1|2|1|1|1|4|1"
    "sn4_megatron_tp2_pp2_vllm_tp4|4|1|2|2|1|1|4|1"
    "sn4_megatron_tp4_vllm_tp4|4|1|4|1|1|1|4|1"
    "sn4_megatron_pp4_vllm_tp4|4|1|1|4|1|1|4|1"
  )
fi

if [ "${MATRIX_NODES}" -ge 2 ]; then
  CASES+=(
    "mn2x2_megatron_dp4_vllm_tp2|2|2|1|1|1|1|2|1"
    "mn2x2_megatron_tp2_vllm_tp2|2|2|2|1|1|1|2|1"
    "mn2x2_vllm_tp4_cross_node|2|2|1|1|1|1|4|1"
  )
fi

if [ ! -f "${RESULT_MD}" ]; then
  cat > "${RESULT_MD}" <<EOF
# AMD parallelism matrix

Updated automatically by \`run_parallelism_matrix.sh\`.

| Case | GPUs/node | Nodes | Megatron TP/PP/CP/EP | vLLM engines x TP | Result | Log |
|------|-----------|-------|----------------------|-------------------|--------|-----|
EOF
fi

run_case() {
  local spec="$1"
  IFS='|' read -r name gpus nodes mtp mpp mcp mep vtp neng <<<"${spec}"
  if [ -n "${FILTER}" ] && [[ "${name}" != *"${FILTER}"* ]]; then
    echo "SKIP ${name} (filter=${FILTER})"
    return 0
  fi
  if [ "${gpus}" -gt "${MATRIX_GPUS}" ] && [ "${nodes}" -eq 1 ]; then
    echo "SKIP ${name} (needs ${gpus} GPUs, have ${MATRIX_GPUS})"
    return 0
  fi

  local log="${REPORT_DIR}/matrix_${name}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "[$(ts)] RUN ${name} -> ${log}"
  set +e
  NUM_GPUS="${gpus}" \
  NUM_NODES="${nodes}" \
  MEGATRON_TP="${mtp}" \
  MEGATRON_PP="${mpp}" \
  MEGATRON_CP="${mcp}" \
  MEGATRON_EP="${mep}" \
  VLLM_TP="${vtp}" \
  NUM_ENGINES="${neng}" \
    bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh \
    > "${log}" 2>&1
  local rc=$?
  set -e
  local result="FAIL rc=${rc}"
  if grep -q "Training done!" "${log}" 2>/dev/null; then
    result="PASS"
  elif grep -qi "Traceback\|Error\|FAILED" "${log}" 2>/dev/null; then
    result="FAIL rc=${rc}"
  fi
  echo "[$(ts)] ${result} ${name}"
  printf '| `%s` | %s | %s | %s/%s/%s/%s | %sx%s | %s | `%s` |\n' \
    "${name}" "${gpus}" "${nodes}" "${mtp}" "${mpp}" "${mcp}" "${mep}" \
    "${neng}" "${vtp}" "${result}" "$(basename "${log}")" >> "${RESULT_MD}"
  bash integrations/rocm_amd/gpu_cleanup.sh >/dev/null 2>&1 || true
}

for spec in "${CASES[@]}"; do
  run_case "${spec}"
done

echo "Matrix written to ${RESULT_MD}"
