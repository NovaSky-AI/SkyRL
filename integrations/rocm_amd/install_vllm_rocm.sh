#!/usr/bin/env bash
# Install vLLM for ROCm without upgrading PyTorch.
# Prefer a wheel built with build_vllm_rocm.sh (cached under .vllm_rocm_cache/wheels/).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_CACHE="${SCRIPT_DIR}/.vllm_rocm_cache/wheels"
VERIFY_PY="${SCRIPT_DIR}/verify_vllm_skyrl_compat.py"

mkdir -p "${WHEEL_CACHE}"
# Drop legacy ROCm wheels (<0.20) that predate SkyRL's HTTP inference server APIs.
find "${WHEEL_CACHE}" -maxdepth 1 -name 'vllm-*.whl' ! -name 'vllm-0.20*.whl' -delete 2>/dev/null || true

TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"
echo "Current torch ${TORCH_VER}"

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
