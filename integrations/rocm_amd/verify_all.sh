#!/usr/bin/env bash
# Run static checks + ROCm integration tests (inside GPU container with stack installed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SCRIPT_DIR="${ROOT}/integrations/rocm_amd"

echo "=== Static: shell syntax ==="
for sh in "${SCRIPT_DIR}"/*.sh; do
  bash -n "$sh"
done

echo "=== Static: python compile ==="
python3 -m py_compile \
  "${SCRIPT_DIR}/gpu_support.py" \
  "${SCRIPT_DIR}/ray_preflight.py" \
  "${SCRIPT_DIR}/smoke_test.py" \
  "${SCRIPT_DIR}/verify_vllm_skyrl_compat.py"

echo "=== GPU support ==="
python3 "${SCRIPT_DIR}/gpu_support.py"

echo "=== Install full stack (cached vLLM wheel if present) ==="
bash "${SCRIPT_DIR}/install_full_stack.sh"

echo "=== Ray preflight (1 GPU) ==="
export NUM_GPUS=1
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
bash "${SCRIPT_DIR}/ray_preflight.sh"
bash "${SCRIPT_DIR}/gpu_cleanup.sh"

echo "=== vLLM SkyRL compat ==="
python3 "${SCRIPT_DIR}/verify_vllm_skyrl_compat.py"

echo "=== Megatron ROCm validate (1 GPU, before vLLM smoke) ==="
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
bash "${SCRIPT_DIR}/validate_megatron_rocm.sh"
bash "${SCRIPT_DIR}/gpu_cleanup.sh"

echo "=== vLLM LLM smoke (single GPU) ==="
export VLLM_SMOKE_GPU=0
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
bash "${SCRIPT_DIR}/vllm_llm_smoke.sh"
bash "${SCRIPT_DIR}/gpu_cleanup.sh"

echo "=== Integration smoke test ==="
export NUM_GPUS=1
bash "${SCRIPT_DIR}/ray_preflight.sh"
bash "${SCRIPT_DIR}/run_smoke_test.sh"
bash "${SCRIPT_DIR}/gpu_cleanup.sh"

if [ "${RUN_GRPO_SMOKE:-0}" = "1" ]; then
  echo "=== GRPO smoke (GSM8K, 2 GPUs) ==="
  export NUM_GPUS=2
  export HIP_VISIBLE_DEVICES=0,1
  export ROCR_VISIBLE_DEVICES=0,1
  unset CUDA_VISIBLE_DEVICES
  bash "${SCRIPT_DIR}/ray_preflight.sh"
  bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh
fi

echo "=== ALL VERIFY CHECKS PASSED ==="
