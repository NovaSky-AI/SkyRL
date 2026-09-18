#!/usr/bin/env bash
# Install vLLM for ROCm without upgrading PyTorch.
# Prefer a wheel built with build_vllm_rocm.sh (cached under .vllm_rocm_cache/wheels/).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_PY="${SCRIPT_DIR}/verify_vllm_skyrl_compat.py"
VLLM_REV="${VLLM_REV:-bc150f50299199599673614f80d12a196f377655}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942;gfx950}"

TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"
HIP_VER="$(python3 -c 'import torch; print(torch.version.hip or "none")')"
PYTHON_SOABI="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')"
CACHE_KEY_INPUT="vllm=${VLLM_REV}|torch=${TORCH_VER}|hip=${HIP_VER}|arch=${PYTORCH_ROCM_ARCH}|python=${PYTHON_SOABI}"
CACHE_KEY="$(printf '%s' "${CACHE_KEY_INPUT}" | sha256sum | cut -d' ' -f1)"
WHEEL_CACHE="${SCRIPT_DIR}/.vllm_rocm_cache/wheels/${CACHE_KEY}"

mkdir -p "${WHEEL_CACHE}"
echo "Current torch ${TORCH_VER}"
echo "vLLM wheel cache key: ${CACHE_KEY_INPUT}"

install_runtime_deps() {
  python3 -m pip install --no-cache-dir -q \
    -r "${SCRIPT_DIR}/vllm_rocm_runtime_requirements.txt"
}

assert_torch_unchanged() {
  local after
  after="$(python3 -c 'import torch; print(torch.__version__)')"
  if [ "$after" != "$TORCH_VER" ]; then
    echo "ERROR: torch changed ${TORCH_VER} -> ${after}; refusing to continue"
    exit 1
  fi
}

try_cached_wheel() {
  local wheel="$1"
  echo "Installing cached wheel ${wheel}"
  python3 -m pip install --no-cache-dir --force-reinstall --no-deps "${wheel}"
  install_runtime_deps
  assert_torch_unchanged
  python3 "${VERIFY_PY}" || return 1
}

WHEEL=""
if compgen -G "${WHEEL_CACHE}/vllm-0.20*.whl" >/dev/null; then
  WHEEL="$(ls -1t "${WHEEL_CACHE}"/vllm-0.20*.whl | head -1)"
elif compgen -G "${WHEEL_CACHE}/vllm-*.whl" >/dev/null; then
  WHEEL="$(ls -1t "${WHEEL_CACHE}"/vllm-*.whl | head -1)"
fi

if [ -n "${WHEEL}" ]; then
  if try_cached_wheel "${WHEEL}"; then
    python3 -c "import vllm; print('vllm', vllm.__version__, 'torch', __import__('torch').__version__)"
    exit 0
  fi
  echo "Stale or incompatible cached wheel ${WHEEL}; rebuilding"
  rm -f "${WHEEL}"
fi

if python3 "${VERIFY_PY}" 2>/dev/null; then
  echo "vLLM already installed and SkyRL-compatible"
  python3 "${VERIFY_PY}"
  exit 0
fi

bash "${SCRIPT_DIR}/build_vllm_rocm.sh"
