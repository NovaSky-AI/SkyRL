#!/usr/bin/env bash
# Build the SkyRL ROCm Megatron image.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-skyrl-rocm-megatron}"
ROCM_BASE_IMAGE="${ROCM_BASE_IMAGE:-rocm/primus:v26.4}"
BUILDER="${BUILDER:-docker}"

if ! command -v "$BUILDER" >/dev/null 2>&1; then
  if command -v podman >/dev/null 2>&1; then
    BUILDER=podman
  else
    echo "Neither docker nor podman found." >&2
    exit 1
  fi
fi

echo "Building ${IMAGE} from ${ROCM_BASE_IMAGE} using ${BUILDER} ..."
"$BUILDER" build \
  -f docker/Dockerfile.rocm-megatron \
  --build-arg "ROCM_BASE_IMAGE=${ROCM_BASE_IMAGE}" \
  -t "$IMAGE" \
  .

echo
echo "Built ${IMAGE}"
echo "Run with repo bind-mount:"
echo "  bash integrations/rocm_amd/run_in_container.sh"
