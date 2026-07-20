#!/usr/bin/env bash
set -x

# Non-colocated DAPO training+generation for Qwen3.5-35B-A3B-Base using
# checkpoint-delta weight sync. Defaults target 2 nodes:
#   - 1 trainer node with 8 GPUs
#   - 1 inference node with 1 vLLM engine at TP=8
#
# Prepare DAPO data first:
#   bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#
# Example:
#   SYNC_DIR=gs://<bucket>/<prefix>/$(date +%Y%m%d_%H%M%S) \
#   bash examples/train/delta_weight_sync/run_dapo_qwen3.5_35b_a3b_delta.sh

: "${MODEL_NAME:=Qwen/Qwen3.5-35B-A3B-Base}"
: "${DATA_DIR:=$HOME/data/dapo}"
: "${TRAIN_FILE:=$DATA_DIR/dapo-math-17k-cleaned.parquet}"
: "${TEST_FILE:=$DATA_DIR/aime-2024-cleaned.parquet}"
: "${TRAINER_NUM_NODES:=1}"
: "${TRAINER_GPUS_PER_NODE:=8}"
: "${NUM_INFERENCE_ENGINES:=1}"
: "${INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE:=8}"
: "${LOGGER:=wandb}"  # change to "console" to print to stdout
: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${RUN_NAME:=dapo-qwen35b-a3b-delta-${RUN_ID}}"
: "${SYNC_DIR:?Set SYNC_DIR to a unique gs:// or shared filesystem path for this run}"
: "${LOCAL_CHECKPOINT_DIR:=/tmp/skyrl-delta-checkpoints/${RUN_NAME}}"
: "${PUBLISH_STAGING_DIR:=}"
: "${MAX_FILE_SIZE_IN_GB:=2}"
: "${GCS_DOWNLOAD_WORKERS:=4}"
: "${PUBLISH_NUM_WORKERS:=8}"
: "${CHECKPOINT_LOAD_FORMAT:=vllm_fastsafetensors}"
: "${MULTI_THREAD_SAFETENSORS_MAX_WORKERS:=8}"
: "${RAY_DEDUP_LOGS:=0}"

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))

TRAIN_BATCH_SIZE=128
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=16
EVAL_N_SAMPLES_PER_PROMPT=32
ENFORCE_EAGER=true
LR=1e-6

# Megatron config from the colocated DAPO recipe.
MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

TIS_IMP_RATIO_CAP=2.0
TIS_TYPE=token

# Override these for memory-constrained trainer nodes.
OPTIMIZER_OFFLOAD=false
OPTIMIZER_OFFLOAD_FRACTION=0.0

# Qwen3.5 flags.
LANGUAGE_MODEL_ONLY=True
ENGINE_INIT_KWARGS='{"gdn_prefill_backend": "triton", "kernel_config": {"moe_backend": "triton"}}'
DISTRIBUTED_EXECUTOR_BACKEND="mp"
: "${GPU_MEMORY_UTILIZATION:=0.7}"

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

SKYRL_DUMP_INFRA_LOG_TO_STDOUT="${SKYRL_DUMP_INFRA_LOG_TO_STDOUT:-1}" \
RAY_DEDUP_LOGS="$RAY_DEDUP_LOGS" \
uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$TRAINER_NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$TRAINER_GPUS_PER_NODE \
  trainer.placement.ref_num_nodes=$TRAINER_NUM_NODES \
  trainer.placement.ref_num_gpus_per_node=$TRAINER_GPUS_PER_NODE \
  generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.epochs=20 \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=40 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=delta \
  generator.inference_engine.delta_weight_sync.sync_dir="$SYNC_DIR" \
  generator.inference_engine.delta_weight_sync.local_checkpoint_dir="$LOCAL_CHECKPOINT_DIR" \
  generator.inference_engine.delta_weight_sync.publish_staging_dir="$PUBLISH_STAGING_DIR" \
  generator.inference_engine.delta_weight_sync.max_file_size_in_gb=$MAX_FILE_SIZE_IN_GB \
  generator.inference_engine.delta_weight_sync.gcs_download_workers=$GCS_DOWNLOAD_WORKERS \
  generator.inference_engine.delta_weight_sync.publish_num_workers="$PUBLISH_NUM_WORKERS" \
  generator.inference_engine.delta_weight_sync.checkpoint_load_format="$CHECKPOINT_LOAD_FORMAT" \
  generator.inference_engine.delta_weight_sync.multi_thread_safetensors_max_workers=$MULTI_THREAD_SAFETENSORS_MAX_WORKERS \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen3_5_dapo_delta_weight_sync" \
  trainer.run_name="$RUN_NAME" \
  trainer.export_path="$HOME/exports/${RUN_NAME}" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.log_path="/tmp/skyrl-logs-${RUN_NAME}" \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  "$@"
