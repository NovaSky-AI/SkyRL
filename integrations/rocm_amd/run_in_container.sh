#!/usr/bin/env bash
# Run a command inside a ROCm container with SkyRL bind-mounted.
#
# Examples:
#   bash integrations/rocm_amd/run_in_container.sh
#   bash integrations/rocm_amd/run_in_container.sh bash integrations/rocm_amd/install_full_stack.sh
#   bash integrations/rocm_amd/run_in_container.sh bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The default image is tested on the supported CDNA3 and CDNA4 GPUs.
# Substitute only images whose PyTorch/TE builds include the target GPU ISA.
ROCM_IMAGE="${ROCM_IMAGE:-rocm/primus:v26.4}"
BUILDER="${BUILDER:-docker}"

if ! command -v "${BUILDER}" >/dev/null 2>&1; then
  if command -v podman >/dev/null 2>&1; then
    BUILDER=podman
  else
    echo "Neither docker nor podman found." >&2
    exit 1
  fi
fi

GPUS="${ROCM_GPUS:-${NUM_GPUS:-1}}"
if [ "${GPUS}" -eq 1 ]; then
  DEV_LIST="0"
else
  DEV_LIST="$(python3 -c "print(','.join(str(i) for i in range(${GPUS})))")"
fi

if [ "$#" -eq 0 ]; then
  set -- bash
fi

TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-it)
fi

"${BUILDER}" run --rm "${TTY_ARGS[@]}" --network host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES="${DEV_LIST}" \
  -e ROCR_VISIBLE_DEVICES="${DEV_LIST}" \
  -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
  -e RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
  -e RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
  -e NUM_GPUS="${GPUS}" \
  -v "${ROOT}:/workspace/SkyRL" \
  -w /workspace/SkyRL \
  "${ROCM_IMAGE}" \
  "$@"
