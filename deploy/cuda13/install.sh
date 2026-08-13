#!/usr/bin/env bash
# CUDA-13 (Blackwell-Ultra / sm103, e.g. B300) install for SkyRL's megatron
# stack, from the pinned export in requirements-megatron.txt.
#
# Requirements:
#   - NVIDIA driver >= R580 (CUDA 13 user-space needs it).
#   - CUDA_HOME pointing at a CUDA >= 13.0 toolkit (transformer-engine-torch,
#     causal-conv1d, mamba-ssm and nv-grouped-gemm compile from sdist against
#     the runtime torch, which is the CUDA-13 build on PyPI for torch 2.11).
#   - Run from the repository root:  bash deploy/cuda13/install.sh
#
# The target venv defaults to .venv; override with SKYRL_CUDA13_VENV.
set -euo pipefail

ROOT="$(pwd)"
VENV="${SKYRL_CUDA13_VENV:-$ROOT/.venv}"
REQ="$ROOT/deploy/cuda13/requirements-megatron.txt"

[ -f "$REQ" ] || { echo "run from the repository root (missing $REQ)" >&2; exit 1; }
[ -n "${CUDA_HOME:-}" ] || { echo "set CUDA_HOME to a CUDA >= 13.0 toolkit" >&2; exit 1; }

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

uv venv "$VENV" --python 3.12 --allow-existing
PY="$VENV/bin/python"

# All uv pip invocations run from a neutral directory: uv pip picks up
# tool.uv build settings from a pyproject in the working directory, and the
# root project's settings (match-runtime build dependencies) only apply to
# `uv sync` style builds.
cd "$BUILD_DIR"

# Phase 1: the packages that must be importable while the sdists below build
# (they are also part of the pinned set, so versions stay consistent).
uv pip install --python "$PY" torch==2.11.0 ninja packaging setuptools pybind11

# Phase 2a: transformer-engine-torch. Its sdist imports an in-tree
# `build_tools` helper package that only resolves when building from the
# extracted source tree, so install it from a checkout of the sdist.
TE_TORCH_VERSION="$(sed -n 's/^transformer-engine-torch==\([0-9.]*\).*/\1/p' "$REQ" | head -1)"
SDIST_URL="$(curl -sL "https://pypi.org/pypi/transformer-engine-torch/$TE_TORCH_VERSION/json" \
    | "$PY" -c "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['packagetype']=='sdist'][0])")"
curl -sL "$SDIST_URL" | tar xz
uv pip install --python "$PY" --no-build-isolation \
    --override "$ROOT/deploy/cuda13/overrides.txt" \
    "./transformer_engine_torch-$TE_TORCH_VERSION"

# Phase 2b: the full pinned stack. --no-build-isolation lets the remaining
# sdist builds (causal-conv1d, mamba-ssm, nv-grouped-gemm) see the phase-1
# torch, matching the root project's no-build-isolation-package.
uv pip install --python "$PY" --no-build-isolation \
    --override "$ROOT/deploy/cuda13/overrides.txt" -r "$REQ"

# Phase 3: SkyRL itself (and its workspace gym package), editable, with
# dependencies already satisfied above.
uv pip install --python "$PY" --no-deps -e "$ROOT" -e "$ROOT/skyrl-gym"

echo "CUDA-13 stack installed into $VENV"
