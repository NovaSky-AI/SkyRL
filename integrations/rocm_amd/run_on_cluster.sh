#!/usr/bin/env bash
# Slurm launcher: verify stack or run end-to-end Megatron GRPO on AMD GPUs.
#
# Examples:
#   bash integrations/rocm_amd/run_on_cluster.sh verify
#   bash integrations/rocm_amd/run_on_cluster.sh grpo
#   bash integrations/rocm_amd/run_on_cluster.sh e2e   # verify + GRPO smoke
#   RUN_GRPO_SMOKE=1 bash integrations/rocm_amd/run_on_cluster.sh verify
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-verify}"
shift || true

IMAGE="${ROCM_IMAGE:-rocm/primus:v26.4}"
GPUS="${SLURM_GPUS:-2}"
GPU_DEVS="$(python3 -c "print(','.join(str(i) for i in range(${GPUS})))")"
CPUS="${SLURM_CPUS:-16}"
MEM="${SLURM_MEM:-128G}"
TIME="${SLURM_TIME:-02:00:00}"
PARTITION="${SLURM_PARTITION:-rad}"
LOG="${ROOT}/integrations/rocm_amd/reports/cluster_${MODE}_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "${ROOT}/integrations/rocm_amd/reports"

case "${MODE}" in
  verify)
    INNER='bash integrations/rocm_amd/verify_all.sh'
    ;;
  grpo)
    INNER='bash integrations/rocm_amd/install_full_stack.sh && NUM_GPUS='"${GPUS}"' bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh'
    ;;
  e2e)
    INNER='export RUN_GRPO_SMOKE=1 && bash integrations/rocm_amd/verify_all.sh'
    GPUS="${SLURM_GPUS:-2}"
    GPU_DEVS="$(python3 -c "print(','.join(str(i) for i in range(${GPUS})))")"
    ;;
  *)
    echo "Usage: $0 {verify|grpo|e2e} [extra args passed to srun]" >&2
    exit 1
    ;;
esac

echo "Cluster ${MODE} -> ${LOG}"

srun -p "${PARTITION}" \
  --gres="gpu:${GPUS}" \
  -c "${CPUS}" \
  --mem="${MEM}" \
  -t "${TIME}" \
  "$@" \
  bash -lc "
set -euo pipefail
export HIP_VISIBLE_DEVICES=${GPU_DEVS}
export ROCR_VISIBLE_DEVICES=${GPU_DEVS}
export NUM_GPUS=${GPUS}
export MAX_JOBS=\${MAX_JOBS:-16}
export RUN_GRPO_SMOKE=\${RUN_GRPO_SMOKE:-0}

docker run --rm --network host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=${GPU_DEVS} \
  -e ROCR_VISIBLE_DEVICES=${GPU_DEVS} \
  -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
  -e RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
  -e RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
  -e NUM_GPUS -e MAX_JOBS=16 -e RUN_GRPO_SMOKE -e FORCE_VLLM_REBUILD \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -v '${ROOT}:/workspace/SkyRL' \
  -w /workspace/SkyRL \
  '${IMAGE}' \
  bash -lc '${INNER}'
" 2>&1 | tee "${LOG}"

echo "Done. Log: ${LOG}"
