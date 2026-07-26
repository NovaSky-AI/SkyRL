#!/usr/bin/env bash
set -x

# Disaggregated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K
# using checkpoint-delta weight sync through S3.
# Ensure that you populate the `EXTERNAL_PROXY_URL` and `EXTERNAL_SERVER_URLS`


: "${EXTERNAL_PROXY_URL:?Set EXTERNAL_PROXY_URL to the proxy url launched with the serve entrypoint}"
: "${EXTERNAL_SERVER_URLS:?Set EXTERNAL_SERVER_URLS to the server urls of the individual servers}"

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${TRAINER_NUM_GPUS:=4}"
: "${NUM_INFERENCE_ENGINES:=4}"
: "${INFERENCE_TP_SIZE:=1}"
: "${LOGGER:=wandb}"
: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${RUN_NAME:=gsm8k-qwen1p5b-delta-s3-disagg-${RUN_ID}}"
: "${SYNC_DIR:?Set SYNC_DIR to a unique s3:// path for this run}"
: "${LOCAL_CHECKPOINT_DIR:=/tmp/skyrl-delta-checkpoints/${RUN_NAME}}"
: "${PUBLISH_STAGING_DIR:=}"
: "${MAX_TRAINING_STEPS:=20}"
: "${MAX_FILE_SIZE_IN_GB:=1}"
: "${CLOUD_DOWNLOAD_WORKERS:=4}"
: "${PUBLISH_NUM_WORKERS:=8}"
# Cloud CLI used to move delta payloads. gs:// needs the `gcloud` CLI on the node;
# s3:// needs `s5cmd`, which the `aws` extra installs into the run's venv.
: "${CLOUD_EXTRA:=aws}"

SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 \
uv run --isolated --extra fsdp --extra "$CLOUD_EXTRA" -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=$TRAINER_NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$TRAINER_NUM_GPUS \
  trainer.strategy=fsdp \
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
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.external_proxy_url=$EXTERNAL_PROXY_URL \
  generator.inference_engine.external_server_urls="$EXTERNAL_SERVER_URLS" \
  generator.inference_engine.weight_sync_backend=delta \
  generator.inference_engine.delta_weight_sync.sync_dir="$SYNC_DIR" \
  generator.inference_engine.delta_weight_sync.local_checkpoint_dir="$LOCAL_CHECKPOINT_DIR" \
  generator.inference_engine.delta_weight_sync.publish_staging_dir="$PUBLISH_STAGING_DIR" \
  generator.inference_engine.delta_weight_sync.max_file_size_in_gb=$MAX_FILE_SIZE_IN_GB \
  generator.inference_engine.delta_weight_sync.cloud_download_workers=$CLOUD_DOWNLOAD_WORKERS \
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
