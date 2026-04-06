set -ex

# wandb api key.
# export WANDB_API_KEY=YOUR_KEY_HERE

# Pick the sandbox provider and provide the credentials.
# export DAYTONA_API_KEY=YOUR_KEY_HERE

#-----------------------
# Dataset setup
#-----------------------
DATA_DIR="$HOME/data/harbor"
TRAIN_DATA="['$DATA_DIR/CodeContests']"

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="codecontest-stepwise"
TRIALS_DIR="$HOME/$RUN_NAME/trials_run"
CKPTS_DIR="$HOME/$RUN_NAME/ckpts"
EXPORTS_DIR="/mnt/local_storage/$RUN_NAME/exports"
LOG_DIR="/tmp/skyrl-logs/$RUN_NAME"

#-----------------------
# Training setup
#-----------------------
MINI_BATCH_SIZE=32
MAX_MODEL_LEN=32768
APPLY_OVERLONG_FILTERING=true

# Dr. GRPO parameters
LOSS_REDUCTION="seq_mean_token_sum_norm"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false

#----------------
# Infrastructure setup
#----------------
NUM_GPUS=8
ENABLE_RATE_LIMITING=true
MAX_CONCURRENCY=500

# Reduce CUDA memory fragmentation — may help avoid OOM during weight sync
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Prevent FSDP2 async collectives from pinning memory with record_stream
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Run SkyRL command
uv run --isolated --extra fsdp --extra harbor -m examples.train_integrations.harbor.entrypoints.main_harbor \
  data.train_data=$TRAIN_DATA \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=Qwen3-8B \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.policy_loss_type=dual_clip \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  trainer.epochs=3 \
  trainer.eval_before_train=false \
  trainer.eval_interval=900 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=2 \
  trainer.hf_save_interval=2 \
  trainer.max_ckpts_to_keep=3 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.n_samples_per_prompt=8 \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger=wandb \
  trainer.project_name=harbor \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host=127.0.0.1 \
  generator.inference_engine.http_endpoint_port=8000 \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  generator.step_wise_trajectories=true \
  "$@"
