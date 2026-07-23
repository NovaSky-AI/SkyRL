#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export HOME="${SKYRL_HOME:-/tmp}"
export SKYRL_RAY_NUM_CPUS="${SKYRL_RAY_NUM_CPUS:-64}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/usr}"

UV_ARGS=(--active --no-sync --extra tinker --extra fsdp)

LOG_DIR="${TINKER_API_LOG_DIR:-logs/tinker-api}"
if [[ -n "${TINKER_API_LOG_FILE:-}" ]]; then
    LOG_FILE="${TINKER_API_LOG_FILE}"
    mkdir -p "$(dirname "${LOG_FILE}")"
else
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/tinker-api-amd-$(date +%Y%m%d-%H%M%S).log"
fi

write_log_header() {
    {
        printf 'SkyRL AMD Tinker API log\n'
        printf 'started_at=%s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)"
        printf 'cwd=%s\n' "$(pwd)"
        printf 'HOME=%s\n' "${HOME}"
        printf 'SKYRL_RAY_NUM_CPUS=%s\n' "${SKYRL_RAY_NUM_CPUS}"
        printf 'UV_PROJECT_ENVIRONMENT=%s\n' "${UV_PROJECT_ENVIRONMENT}"
        printf 'log_file=%s\n' "${LOG_FILE}"
        printf 'command=uv run'
        printf ' %q' "${UV_ARGS[@]}" -m skyrl.tinker.api "$@"
        printf '\n\n'
    } | tee "${LOG_FILE}"
}

run_api() {
    write_log_header "$@"

    set +e
    uv run "${UV_ARGS[@]}" -m skyrl.tinker.api "$@" 2>&1 | tee -a "${LOG_FILE}"
    status=${PIPESTATUS[0]}

    {
        printf '\nfinished_at=%s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)"
        printf 'exit_status=%s\n' "${status}"
    } | tee -a "${LOG_FILE}"

    set -e
    return "${status}"
}

if [[ "$#" -gt 0 ]]; then
    run_api "$@"
    exit "$?"
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
BACKEND="${BACKEND:-fsdp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"

POLICY_NUM_NODES="${POLICY_NUM_NODES:-1}"
POLICY_NUM_GPUS_PER_NODE="${POLICY_NUM_GPUS_PER_NODE:-1}"
COLOCATE_ALL="${COLOCATE_ALL:-false}"
MICRO_TRAIN_BATCH_SIZE_PER_GPU="${MICRO_TRAIN_BATCH_SIZE_PER_GPU:-32}"
MICRO_FORWARD_BATCH_SIZE_PER_GPU="${MICRO_FORWARD_BATCH_SIZE_PER_GPU:-32}"

INFERENCE_NUM_ENGINES="${INFERENCE_NUM_ENGINES:-6}"
INFERENCE_TENSOR_PARALLEL_SIZE="${INFERENCE_TENSOR_PARALLEL_SIZE:-1}"
INFERENCE_MAX_NUM_BATCHED_TOKENS="${INFERENCE_MAX_NUM_BATCHED_TOKENS:-32768}"
INFERENCE_MAX_NUM_SEQS="${INFERENCE_MAX_NUM_SEQS:-256}"
INFERENCE_MAX_MODEL_LEN="${INFERENCE_MAX_MODEL_LEN:-32768}"
INFERENCE_GPU_MEMORY_UTILIZATION="${INFERENCE_GPU_MEMORY_UTILIZATION:-0.8}"
INFERENCE_ENABLE_RAY_PROMETHEUS_STATS="${INFERENCE_ENABLE_RAY_PROMETHEUS_STATS:-false}"

BACKEND_CONFIG="${BACKEND_CONFIG:-$(cat <<JSON
{
    "trainer.placement.colocate_all": ${COLOCATE_ALL},
    "trainer.placement.policy_num_nodes": ${POLICY_NUM_NODES},
    "trainer.placement.policy_num_gpus_per_node": ${POLICY_NUM_GPUS_PER_NODE},
    "trainer.micro_train_batch_size_per_gpu": ${MICRO_TRAIN_BATCH_SIZE_PER_GPU},
    "trainer.micro_forward_batch_size_per_gpu": ${MICRO_FORWARD_BATCH_SIZE_PER_GPU},

    "generator.inference_engine.num_engines": ${INFERENCE_NUM_ENGINES},
    "generator.inference_engine.tensor_parallel_size": ${INFERENCE_TENSOR_PARALLEL_SIZE},
    "generator.inference_engine.max_num_batched_tokens": ${INFERENCE_MAX_NUM_BATCHED_TOKENS},
    "generator.inference_engine.enable_ray_prometheus_stats": ${INFERENCE_ENABLE_RAY_PROMETHEUS_STATS},
    "generator.inference_engine.gpu_memory_utilization": ${INFERENCE_GPU_MEMORY_UTILIZATION},
    "generator.inference_engine.max_num_seqs": ${INFERENCE_MAX_NUM_SEQS},
    "generator.inference_engine.engine_init_kwargs.max_model_len": ${INFERENCE_MAX_MODEL_LEN}
}
JSON
)}"

run_api \
    --base-model "${BASE_MODEL}" \
    --backend "${BACKEND}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --backend-config "${BACKEND_CONFIG}"
