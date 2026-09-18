#!/usr/bin/env bash
# Full ROCm stack: Megatron-Bridge + vLLM (ROCm build) + SkyRL for training + inference on AMD GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/install_megatron_bridge.sh"

if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
  echo "--- verify supported Instinct GPU (MI300X/MI325X/MI355X) ---"
  python3 "${SCRIPT_DIR}/gpu_support.py" || {
    echo "WARN: GPU not supported or not detected; see integrations/rocm_amd/README.md#supported-gpus"
    exit 1
  }
else
  echo "SKIP: no visible GPU during install (OK for image build without devices)"
fi

echo "--- install vLLM (ROCm, SkyRL 0.20.x API) ---"
bash "${SCRIPT_DIR}/install_vllm_rocm.sh"

echo "--- ensure Ray 2.57 (SkyRL pin) ---"
python3 -m pip install --no-cache-dir --force-reinstall "ray[default]==2.57.0"

echo "--- re-pin transformers for SkyRL after vLLM deps ---"
python3 -m pip install --no-cache-dir --force-reinstall "transformers>=5.6.1,<=5.8.0"

echo "--- verify Ray scheduling API ---"
python3 -c "
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
print('ray', ray.__version__, 'PlacementGroupSchedulingStrategy OK')
"

echo "=== Full ROCm AMD stack ready ==="
python3 -c "
import torch
from megatron.bridge import AutoBridge
import megatron.core as mc
print('torch', torch.__version__, 'hip', getattr(torch.version,'hip',None))
print('megatron-core', mc.__version__)
print('AutoBridge', AutoBridge)
import importlib.util
print('vllm', bool(importlib.util.find_spec('vllm')))
"
python3 "${SCRIPT_DIR}/verify_vllm_skyrl_compat.py"
