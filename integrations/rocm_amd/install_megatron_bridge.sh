#!/usr/bin/env bash
# Install Megatron-Bridge + megatron-core 0.19 and SkyRL on ROCm (inside rocm/primus or skyrl-rocm-megatron image).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MCORE_REV="${MCORE_REV:-71e418ea7d7b3a6c9a53238c543c3e0b43e11026}"
BRIDGE_REV="${BRIDGE_REV:-91a15142a4b4442a8d46ab539d1b923bd08570d0}"

# Some ROCm images ship legacy Megatron-LM on PYTHONPATH; SkyRL needs megatron-core 0.19 from git.
export PYTHONPATH="$(python3 - <<'PY'
import os
print(":".join(p for p in os.environ.get("PYTHONPATH", "").split(":") if p and "Megatron-LM" not in p))
PY
)"

export NVTE_USE_ROCM="${NVTE_USE_ROCM:-1}"
export NVTE_USE_HIPBLASLT="${NVTE_USE_HIPBLASLT:-1}"
# Do not set NVTE_FUSED_ATTN=0 on ROCm — breaks Megatron-Bridge model materialize.

echo "Installing megatron-core @ ${MCORE_REV}"
python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
  "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM@${MCORE_REV}"

echo "Installing bridge deps + nvidia-modelopt"
python3 -m pip install --no-cache-dir \
  nvidia-modelopt \
  "transformers>=5.6.1,<=5.8.0" \
  "mistral-common>=1.10.0" \
  "peft>=0.18.1" \
  "datasets>=4.0.0" \
  accelerate diffusers einops imageio imageio-ffmpeg \
  "omegaconf>=2.3.0" tensorboard typing-extensions rich wandb six regex pyyaml tqdm \
  "hydra-core>=1.3,<=1.3.2" \
  qwen-vl-utils flash-linear-attention timm \
  "open-clip-torch>=3.2.0"

echo "Installing megatron-bridge @ ${BRIDGE_REV} (--no-deps)"
# setuptools 80+ can fail this source build with FileExistsError on .dist-info vs .egg-info.
python3 -m pip install --no-cache-dir "setuptools>=70,<80"
python3 -m pip install --no-cache-dir --no-deps \
  "megatron-bridge @ git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@${BRIDGE_REV}"

install_skyrl_rocm() (
  local root="$1"
  cd "${root}"
  local restore_pyproject=0
  cleanup() {
    if [ "${restore_pyproject}" -eq 1 ] && [ -f pyproject.toml.skyrl.bak ]; then
      mv pyproject.toml.skyrl.bak pyproject.toml
    fi
  }
  trap cleanup EXIT

  local rocm_pyproject="${root}/docker/pyproject.rocm.toml"
  if [ ! -f "${rocm_pyproject}" ]; then
    echo "ERROR: missing ${rocm_pyproject}" >&2
    echo "Ensure docker/pyproject.rocm.toml exists (integrations/rocm_amd checkout)." >&2
    exit 1
  fi
  if [ -f pyproject.toml ] && ! cmp -s pyproject.toml "${rocm_pyproject}"; then
    cp pyproject.toml pyproject.toml.skyrl.bak
    restore_pyproject=1
  fi
  cp "${rocm_pyproject}" pyproject.toml
  python3 -m pip install --no-cache-dir --force-reinstall "ray[default]==2.57.0"
  python3 -m pip install --no-cache-dir -e ".[rocm-megatron]"
  python3 -m pip install --no-cache-dir "transformers>=5.6.1,<=5.8.0" -q
)

install_skyrl_rocm "${SKYRL_ROOT}"

python3 -c "from megatron.bridge import AutoBridge; import megatron.core as mc; print('OK bridge+mcore', mc.__version__)"
