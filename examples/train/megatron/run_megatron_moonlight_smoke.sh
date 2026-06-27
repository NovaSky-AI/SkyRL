#!/usr/bin/env bash
set -euo pipefail
set -x

# Tiny colocated GRPO training+generation smoke for Moonlight-16B-A3B-Instruct
# with SkyRL's Megatron backend. It uses a four-row in-repo GSM8K-shaped JSONL
# and runs a single training step by default.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DATA_FILE="${DATA_FILE:-${SCRIPT_DIR}/smoke_data/moonlight_gsm8k_smoke.jsonl}"
MODEL_NAME="${MODEL_NAME:-moonshotai/Moonlight-16B-A3B-Instruct}"
LOGGER="${LOGGER:-console}"

NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS="${NUM_GPUS:-4}"

MEGATRON_TP="${MEGATRON_TP:-4}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_EP="${MEGATRON_EP:-4}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"

NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-1}"
INFERENCE_ENGINE_TP="${INFERENCE_ENGINE_TP:-4}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-64}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_A_INIT_METHOD="${LORA_A_INIT_METHOD:-kaiming}"

OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-true}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"

# Moonlight uses MLA. On pre-Hopper GPUs, SkyRL's tests fall back from THD
# sample packing to the unpacked BSHD path, so keep this false for A100 smoke.
REMOVE_MICROBATCH_PADDING="${REMOVE_MICROBATCH_PADDING:-false}"
FLASH_ATTN="${FLASH_ATTN:-false}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
ASYNC_ENGINE="${ASYNC_ENGINE:-false}"
USE_KL_LOSS="${USE_KL_LOSS:-false}"
RESUME_MODE="${RESUME_MODE:-null}"
CKPT_PATH="${CKPT_PATH:-${HOME}/ckpts/moonlight_16b_a3b_smoke}"
RUN_NAME="${RUN_NAME:-moonlight_16b_a3b_megatron_smoke_tp${MEGATRON_TP}_ep${MEGATRON_EP}_lora${LORA_RANK}}"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "Missing DATA_FILE: ${DATA_FILE}" >&2
  exit 2
fi

export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-1}"
export _SKYRL_USE_NEW_INFERENCE="${_SKYRL_USE_NEW_INFERENCE:-0}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

TRAIN_ARGS=(
  "data.train_data=['${DATA_FILE}']"
  "data.val_data=['${DATA_FILE}']"
  "trainer.algorithm.advantage_estimator=grpo"
  "trainer.policy.model.path=${MODEL_NAME}"
  "trainer.placement.colocate_all=true"
  "trainer.strategy=megatron"
  "trainer.placement.policy_num_nodes=${NUM_NODES}"
  "trainer.placement.policy_num_gpus_per_node=${NUM_GPUS}"
  "generator.inference_engine.num_engines=${NUM_INFERENCE_ENGINES}"
  "generator.inference_engine.tensor_parallel_size=${INFERENCE_ENGINE_TP}"
  "generator.inference_engine.distributed_executor_backend=${DISTRIBUTED_EXECUTOR_BACKEND}"
  "trainer.policy.megatron_config.tensor_model_parallel_size=${MEGATRON_TP}"
  "trainer.policy.megatron_config.pipeline_model_parallel_size=${MEGATRON_PP}"
  "trainer.policy.megatron_config.context_parallel_size=${MEGATRON_CP}"
  "trainer.policy.megatron_config.expert_model_parallel_size=${MEGATRON_EP}"
  "trainer.policy.megatron_config.expert_tensor_parallel_size=${MEGATRON_ETP}"
  "trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=${OPTIMIZER_OFFLOAD}"
  "trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=${OPTIMIZER_OFFLOAD}"
  "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=${OPTIMIZER_OFFLOAD}"
  "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}"
  "trainer.remove_microbatch_padding=${REMOVE_MICROBATCH_PADDING}"
  "trainer.flash_attn=${FLASH_ATTN}"
  "trainer.epochs=1"
  "trainer.max_training_steps=${MAX_TRAINING_STEPS}"
  "trainer.eval_batch_size=${TRAIN_BATCH_SIZE}"
  "trainer.eval_before_train=false"
  "trainer.eval_interval=-1"
  "trainer.update_epochs_per_batch=1"
  "trainer.train_batch_size=${TRAIN_BATCH_SIZE}"
  "trainer.policy_mini_batch_size=${POLICY_MINI_BATCH_SIZE}"
  "trainer.micro_forward_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
  "trainer.micro_train_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
  "trainer.ckpt_interval=1000000"
  "trainer.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "generator.sampling_params.max_generate_length=${MAX_GENERATE_LENGTH}"
  "generator.sampling_params.temperature=0.2"
  "generator.sampling_params.top_p=1.0"
  "generator.sampling_params.logprobs=1"
  "trainer.policy.optimizer_config.lr=1.0e-5"
  "trainer.algorithm.use_kl_loss=${USE_KL_LOSS}"
  "generator.inference_engine.backend=${INFERENCE_BACKEND}"
  "generator.inference_engine.run_engines_locally=true"
  "generator.inference_engine.weight_sync_backend=nccl"
  "generator.inference_engine.async_engine=${ASYNC_ENGINE}"
  "generator.batched=true"
  "environment.env_class=gsm8k"
  "generator.n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}"
  "generator.inference_engine.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "trainer.logger=${LOGGER}"
  "trainer.project_name=moonlight_megatron_smoke"
  "trainer.run_name=${RUN_NAME}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.ckpt_path=${CKPT_PATH}"
)

if [[ "${LORA_RANK}" != "0" ]]; then
  TRAIN_ARGS+=(
    "trainer.policy.model.lora.rank=${LORA_RANK}"
    "trainer.policy.model.lora.alpha=${LORA_ALPHA}"
    "trainer.policy.model.lora.init_method=${LORA_A_INIT_METHOD}"
    "trainer.policy.model.lora.target_modules=all-linear"
  )
fi

SKYRL_RAY_PG_TIMEOUT_IN_S="${SKYRL_RAY_PG_TIMEOUT_IN_S:-300}" \
  uv run --isolated --extra megatron --with blobfile \
  -m skyrl.train.entrypoints.main_base \
  "${TRAIN_ARGS[@]}" \
  "$@"
