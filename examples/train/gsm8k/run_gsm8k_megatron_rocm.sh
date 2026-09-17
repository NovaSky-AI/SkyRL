#!/usr/bin/env bash
# Colocated Megatron GRPO on AMD GPUs (GSM8K smoke).
#
# Prerequisites (inside ROCm container):
#   bash integrations/rocm_amd/install_full_stack.sh
#
# Or from the host:
#   bash integrations/rocm_amd/run_in_container.sh bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

: "${DATA_DIR:=$HOME/data/gsm8k_rocm}"
: "${NUM_GPUS:=2}"
: "${MODEL_NAME:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${LOGGER:=console}"
: "${MEGATRON_TP:=1}"
: "${MEGATRON_PP:=1}"
: "${NUM_ENGINES:=${NUM_GPUS}}"
: "${LOG_DIR:=/tmp/skyrl-logs-rocm}"
: "${CKPT_DIR:=$HOME/ckpts/gsm8k_megatron_rocm}"

mkdir -p "$DATA_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "=== Ray + vLLM preflight ==="
bash integrations/rocm_amd/ray_preflight.sh
python3 integrations/rocm_amd/verify_vllm_skyrl_compat.py

echo "=== Preparing GSM8K (tiny subset) ==="
python3 examples/train/gsm8k/gsm8k_dataset.py \
  --output_dir "$DATA_DIR" \
  --max_train_dataset_length 64

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export NVTE_USE_ROCM=1
export NVTE_USE_HIPBLASLT=1
export VLLM_TARGET_DEVICE=rocm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_USE_TRITON_FLASH_ATTN=0
export FLA_TILELANG=0
# Keep PyTorch's CUDA-compatible device mask aligned with ROCm masks.
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
  export ROCR_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
else
  unset CUDA_VISIBLE_DEVICES
fi

echo "=== Starting Megatron GRPO on ROCm (${NUM_GPUS} GPUs) ==="
python3 -m skyrl.train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="${MODEL_NAME}" \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node="${NUM_GPUS}" \
  trainer.placement.ref_num_gpus_per_node="${NUM_GPUS}" \
  trainer.placement.critic_num_gpus_per_node=0 \
  generator.inference_engine.num_engines="${NUM_ENGINES}" \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.policy.megatron_config.tensor_model_parallel_size="${MEGATRON_TP}" \
  trainer.policy.megatron_config.pipeline_model_parallel_size="${MEGATRON_PP}" \
  trainer.ref.megatron_config.tensor_model_parallel_size="${MEGATRON_TP}" \
  trainer.ref.megatron_config.pipeline_model_parallel_size="${MEGATRON_PP}" \
  trainer.remove_microbatch_padding=true \
  trainer.epochs=1 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=256 \
  generator.sampling_params.max_generate_length=128 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=2 \
  generator.inference_engine.engine_init_kwargs.enable_sleep_mode=false \
  generator.inference_engine.engine_init_kwargs.max_model_len=512 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.inference_engine.max_num_batched_tokens=1024 \
  generator.inference_engine.max_num_seqs=32 \
  generator.inference_engine.enforce_eager=true \
  trainer.logger="${LOGGER}" \
  trainer.project_name=gsm8k_megatron_rocm \
  trainer.run_name=rocm_smoke \
  trainer.resume_mode=null \
  trainer.log_path="${LOG_DIR}" \
  trainer.ckpt_path="${CKPT_DIR}" \
  trainer.flash_attn=true \
  "$@"
