#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${CUTLASS_DIR:-}" ]]; then
  echo "CUTLASS_DIR is not set. Point it to your CUTLASS checkout." >&2
  exit 1
fi

if [[ ! -d "${CUTLASS_DIR}" ]]; then
  echo "CUTLASS_DIR does not exist: ${CUTLASS_DIR}" >&2
  exit 1
fi

JAX_INCLUDE_DIR="$(uv run --extra gpu - <<'PY'
import os
import jaxlib
print(os.path.join(os.path.dirname(jaxlib.__file__), "include"))
PY
)"

NVCC_BIN="${NVCC_BIN:-nvcc}"
NVCC_ARCH="${NVCC_ARCH:-sm_90a}"
if ! command -v "${NVCC_BIN}" >/dev/null 2>&1; then
  echo "nvcc not found. Set NVCC_BIN or ensure CUDA is on PATH." >&2
  exit 1
fi

OUT_DIR="${SCRIPT_DIR}/_build"
mkdir -p "${OUT_DIR}"

"${NVCC_BIN}" \
  -O3 \
  -std=c++17 \
  -arch="${NVCC_ARCH}" \
  -shared \
  -Xcompiler -fPIC \
  -I"${JAX_INCLUDE_DIR}" \
  -I"${CUTLASS_DIR}/include" \
  "${SCRIPT_DIR}/ragged_dot_ffi.cu" \
  -o "${OUT_DIR}/libragged_dot_ffi.so"

echo "Built ${OUT_DIR}/libragged_dot_ffi.so"
