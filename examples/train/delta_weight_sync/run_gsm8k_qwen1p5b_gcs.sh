#!/usr/bin/env bash
set -x

# Non-colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K
# using checkpoint-delta weight sync through Google Cloud Storage.

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${TRAINER_NUM_GPUS:=4}"
: "${NUM_INFERENCE_ENGINES:=4}"
: "${INFERENCE_TP_SIZE:=1}"
: "${LOGGER:=wandb}"
: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${RUN_NAME:=gsm8k-qwen1p5b-delta-gcs-${RUN_ID}}"
: "${SYNC_DIR:?Set SYNC_DIR to a unique gs:// path for this run}"
: "${LOCAL_CHECKPOINT_DIR:=/tmp/skyrl-delta-checkpoints/${RUN_NAME}}"
: "${PUBLISH_STAGING_DIR:=}"
: "${MAX_TRAINING_STEPS:=20}"
: "${MAX_FILE_SIZE_IN_GB:=1}"
: "${GCS_DOWNLOAD_WORKERS:=4}"
: "${PUBLISH_NUM_WORKERS:=8}"
: "${RAY_DEDUP_LOGS:=0}"

SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 \
RAY_DEDUP_LOGS="$RAY_DEDUP_LOGS" \
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=$TRAINER_NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$TRAINER_NUM_GPUS \
  trainer.strategy=fsdp \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_TP_SIZE \
  trainer.epochs=20 \
  trainer.max_training_steps=$MAX_TRAINING_STEPS \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=delta \
  generator.inference_engine.delta_weight_sync.sync_dir="$SYNC_DIR" \
  generator.inference_engine.delta_weight_sync.local_checkpoint_dir="$LOCAL_CHECKPOINT_DIR" \
  generator.inference_engine.delta_weight_sync.publish_staging_dir="$PUBLISH_STAGING_DIR" \
  generator.inference_engine.delta_weight_sync.max_file_size_in_gb=$MAX_FILE_SIZE_IN_GB \
  generator.inference_engine.delta_weight_sync.gcs_download_workers=$GCS_DOWNLOAD_WORKERS \
  generator.inference_engine.delta_weight_sync.publish_num_workers="$PUBLISH_NUM_WORKERS" \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k-delta-weight-sync" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs-${RUN_NAME}" \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  "$@"
