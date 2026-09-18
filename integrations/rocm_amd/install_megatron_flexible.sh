#!/usr/bin/env bash
# Install megatron-core + Megatron-Bridge on ROCm without CUDA-only extras.
#
# Override the git revisions with MCORE_REV / BRIDGE_REV (any commit, tag, or
# branch). SkyRL loads Bridge through bridge_compat.py so older/newer HF loader
# names still work. CUDA-only Bridge extras are never installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MCORE_REV="${MCORE_REV:-71e418ea7d7b3a6c9a53238c543c3e0b43e11026}"
BRIDGE_REV="${BRIDGE_REV:-91a15142a4b4442a8d46ab539d1b923bd08570d0}"
MCORE_REPO="${MCORE_REPO:-https://github.com/NVIDIA/Megatron-LM}"
BRIDGE_REPO="${BRIDGE_REPO:-https://github.com/NVIDIA-NeMo/Megatron-Bridge}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=5.6.1,<=5.8.0}"

export PYTHONPATH="$(python3 - <<'PY'
import os
print(":".join(p for p in os.environ.get("PYTHONPATH", "").split(":") if p and "Megatron-LM" not in p))
PY
)"

export NVTE_USE_ROCM="${NVTE_USE_ROCM:-1}"
export NVTE_USE_HIPBLASLT="${NVTE_USE_HIPBLASLT:-1}"

echo "Installing megatron-core @ ${MCORE_REV}"
python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
  "megatron-core @ git+${MCORE_REPO}@${MCORE_REV}"

echo "Installing AMD-safe Bridge runtime deps (${TRANSFORMERS_SPEC})"
python3 -m pip install --no-cache-dir \
  nvidia-modelopt \
  "${TRANSFORMERS_SPEC}" \
  "mistral-common>=1.10.0" \
  "peft>=0.18.1" \
  "datasets>=4.0.0" \
  accelerate diffusers einops imageio imageio-ffmpeg \
  "omegaconf>=2.3.0" tensorboard typing-extensions rich wandb six regex pyyaml tqdm \
  "hydra-core>=1.3,<=1.3.2" \
  qwen-vl-utils flash-linear-attention timm \
  "open-clip-torch>=3.2.0"

echo "Installing megatron-bridge @ ${BRIDGE_REV} (--no-deps)"
python3 -m pip install --no-cache-dir "setuptools>=70,<80"
python3 -m pip install --no-cache-dir --no-deps \
  "megatron-bridge @ git+${BRIDGE_REPO}@${BRIDGE_REV}"

if [ "${SKIP_SKYRL_INSTALL:-0}" != "1" ]; then
  (
    cd "${SKYRL_ROOT}"
    restore_pyproject=0
    cleanup() {
      if [ "${restore_pyproject}" -eq 1 ] && [ -f pyproject.toml.skyrl.bak ]; then
        mv pyproject.toml.skyrl.bak pyproject.toml
      fi
    }
    trap cleanup EXIT
    rocm_pyproject="${SKYRL_ROOT}/docker/pyproject.rocm.toml"
    if [ -f pyproject.toml ] && [ -f "${rocm_pyproject}" ] && ! cmp -s pyproject.toml "${rocm_pyproject}"; then
      cp pyproject.toml pyproject.toml.skyrl.bak
      restore_pyproject=1
      cp "${rocm_pyproject}" pyproject.toml
    fi
    python3 -m pip install --no-cache-dir --force-reinstall "ray[default]==2.57.0"
    python3 -m pip install --no-cache-dir -e ".[rocm-megatron]"
    python3 -m pip install --no-cache-dir "${TRANSFORMERS_SPEC}" -q
  )
fi

python3 "${SCRIPT_DIR}/probe_megatron_compat.py"
python3 -c "from megatron.bridge import AutoBridge; import megatron.core as mc; print('OK bridge+mcore', mc.__version__)"
