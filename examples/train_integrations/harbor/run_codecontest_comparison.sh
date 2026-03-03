set -ex

# Three-way comparison: baseline vs step-wise vs step-wise+TIS
# Usage:
#   ./run_codecontest_comparison.sh baseline
#   ./run_codecontest_comparison.sh stepwise
#   ./run_codecontest_comparison.sh stepwise-tis

MODE="${1:?Usage: $0 <baseline|stepwise|stepwise-tis>}"

#-----------------------
# Dataset setup
#-----------------------
DATA_DIR="$HOME/data/harbor"
TRAIN_DATA="['$DATA_DIR/CodeContests']"

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="codecontest-${MODE}"
TRIALS_DIR="$HOME/$RUN_NAME/trials_run"
CKPTS_DIR="$HOME/$RUN_NAME/ckpts"
EXPORTS_DIR="$HOME/$RUN_NAME/exports"
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

# Chat template for interleaved thinking
CHAT_TEMPLATE_PATH="$(dirname "$0")/../../../skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"

#----------------
# Infrastructure setup
#----------------
NUM_GPUS=8
ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND=5
MAX_CONCURRENCY=512

# Mode-specific settings
STEP_WISE="false"
TIS_RATIO_TYPE="null"
N_SAMPLES=8

case "$MODE" in
  baseline)
    STEP_WISE="false"
    TIS_RATIO_TYPE="null"
    ;;
  stepwise)
    STEP_WISE="true"
    TIS_RATIO_TYPE="null"
    ;;
  stepwise-tis)
    STEP_WISE="true"
    TIS_RATIO_TYPE="token"
    ;;
  *)
    echo "Unknown mode: $MODE. Use: baseline, stepwise, or stepwise-tis"
    exit 1
    ;;
esac

echo "Running mode: $MODE (step_wise=$STEP_WISE, tis=$TIS_RATIO_TYPE)"

# Prevent CUDA OOM fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Kill any lingering Daytona sandboxes first
echo "Cleaning up Daytona sandboxes..."
uv run --isolated --extra harbor python examples/train_integrations/harbor/kill_daytona_sandboxes.py || true

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
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  trainer.epochs=10 \
  trainer.eval_before_train=false \
  trainer.eval_interval=999999 \
  trainer.eval_batch_size=128 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=999999 \
  trainer.hf_save_interval=999999 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.n_samples_per_prompt=$N_SAMPLES \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.step_wise_trajectories=$STEP_WISE \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_RATIO_TYPE \
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
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  "${@:2}"
