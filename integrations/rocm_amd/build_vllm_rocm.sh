#!/usr/bin/env bash
# Build vLLM from source against the container's ROCm PyTorch (do NOT pip install vllm — it pulls CUDA torch).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pin vLLM source to match SkyRL pyproject (vllm==0.20.2). Override with VLLM_REV if needed.
VLLM_TAG="${VLLM_TAG:-v0.20.2}"
VLLM_REV="${VLLM_REV:-bc150f50299199599673614f80d12a196f377655}"
VLLM_SRC="${VLLM_SRC:-/tmp/vllm-rocm-build-${VLLM_REV:0:12}}"
SKYRL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERIFY_PY="${SKYRL_ROOT}/integrations/rocm_amd/verify_vllm_skyrl_compat.py"
CACHE_WHEEL_DIR="${SKYRL_ROOT}/integrations/rocm_amd/.vllm_rocm_cache/wheels"

TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"
echo "Building vLLM ${VLLM_REV} for torch ${TORCH_VER}"
mkdir -p "${CACHE_WHEEL_DIR}"

wheel_is_skyrl_compatible() {
  local wheel="$1"
  python3 -m pip install --no-cache-dir --force-reinstall --no-deps "${wheel}" -q
  python3 "${VERIFY_PY}"
}

if compgen -G "${CACHE_WHEEL_DIR}/vllm-0.20*.whl" >/dev/null; then
  WHEEL="$(ls -1t "${CACHE_WHEEL_DIR}"/vllm-0.20*.whl | head -1)"
fi

if [ -n "${WHEEL:-}" ]; then
  echo "Checking cached wheel ${WHEEL}"
  if wheel_is_skyrl_compatible "${WHEEL}"; then
    python3 -c "
import glob, os, vllm
d = os.path.dirname(vllm.__file__)
so = glob.glob(os.path.join(d, '*.so')) + glob.glob(os.path.join(d, '**', '*.so'), recursive=True)
print('vllm', vllm.__version__, 'native_exts', len(so))
"
    exit 0
  fi
  echo "Stale or incompatible cached wheel; rebuilding at ${VLLM_REV}"
fi

if [ "${FORCE_VLLM_REBUILD:-0}" = "1" ]; then
  echo "FORCE_VLLM_REBUILD=1: removing ${VLLM_SRC} and cached wheels"
  rm -rf "${VLLM_SRC}" "${CACHE_WHEEL_DIR}"/vllm-*.whl
fi

export VLLM_TARGET_DEVICE=rocm
# MI300X/MI325X use gfx942 (CDNA3); MI355X uses gfx950 (CDNA4).
# Build both targets by default so one wheel covers every supported GPU.
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942;gfx950}"
echo "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"
export NINJAFLAGS="-j${MAX_JOBS}"
export VLLM_MAXIMUM_CPU_THREADS="${MAX_JOBS}"

if [ ! -d "${VLLM_SRC}/.git" ]; then
  git clone --filter=blob:none https://github.com/vllm-project/vllm.git "${VLLM_SRC}"
fi
cd "${VLLM_SRC}"
git fetch --depth 1 origin "${VLLM_REV}"
git checkout --detach FETCH_HEAD
git checkout -- csrc/quantization/gptq/compat.cuh

python3 - <<'PY'
from pathlib import Path
import re

path = Path("csrc/quantization/gptq/compat.cuh")
text = path.read_text()
pattern = r"//\n\n#if defined\(__CUDA_ARCH__\).*?#endif\n#endif\n"
replacement = """//

#ifndef USE_ROCM
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ < 700

__device__ __forceinline__ void atomicAdd(half* address, half val) {
  atomicAdd_half(address, val);
}

#if __CUDA_ARCH__ < 600
__device__ __forceinline__ void atomicAdd(half2* address, half2 val) {
  atomicAdd_half2(address, val);
}
#endif

#endif
#endif
#endif

"""
patched, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
if count != 1:
    raise SystemExit(f"unexpected {path} layout; could not patch atomicAdd guard")
path.write_text(patched)
print(f"Patched {path}: CUDA-only atomicAdd fallbacks (ROCm uses native HIP atomics)")
PY

grep -q '#ifndef USE_ROCM' csrc/quantization/gptq/compat.cuh

echo "MAX_JOBS=${MAX_JOBS} (vLLM build parallelism)"
python3 -m pip install --no-cache-dir -q \
  "setuptools==79.0.1" "setuptools-scm==10.2.1" setuptools-rust \
  ninja cmake wheel "packaging<26" pybind11 numba numpy psutil

rm -rf build /tmp/vllm-wheels
mkdir -p /tmp/vllm-wheels
python3 -m pip wheel --no-deps --no-build-isolation -w /tmp/vllm-wheels .

WHEEL="$(ls -1 /tmp/vllm-wheels/vllm-*.whl | tail -1)"
echo "Installing ${WHEEL}"
python3 -m pip install --no-cache-dir --force-reinstall --no-deps "${WHEEL}"

# Cache wheel in repo for faster re-runs.
cp -f "${WHEEL}" "${CACHE_WHEEL_DIR}/"

# Runtime dependencies are explicit so pip never replaces the base image's
# ROCm PyTorch with a CUDA wheel.
python3 -m pip install --no-cache-dir -q \
  -r "${SCRIPT_DIR}/vllm_rocm_runtime_requirements.txt"

AFTER="$(python3 -c 'import torch; print(torch.__version__)')"
if [ "$AFTER" != "$TORCH_VER" ]; then
  echo "ERROR: torch changed during vLLM build ${TORCH_VER} -> ${AFTER}"
  exit 1
fi

cd "${SKYRL_ROOT}"
python3 "${VERIFY_PY}"
python3 -c "
import glob, os, vllm
d = os.path.dirname(vllm.__file__)
so = glob.glob(os.path.join(d, '*.so')) + glob.glob(os.path.join(d, '**', '*.so'), recursive=True)
if not so:
    raise SystemExit(f'vLLM installed but no native extensions under {d}')
print('built vllm', vllm.__version__, 'native_exts', len(so))
"
