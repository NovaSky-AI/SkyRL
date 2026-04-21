set -ex

# wandb api key.
# export WANDB_API_KEY=YOUR_KEY_HERE

# Pick the sandbox provider and provide the credentials.
# export DAYTONA_API_KEY=YOUR_KEY_HERE
# export MODAL_TOKEN_ID=YOUR_KEY_HERE
# export MODAL_TOKEN_SECRET=YOUR_KEY_HERE

# ---- OpenHands-specific env vars ----
# Disable condensation to ensure strictly-appending chat history for RL.
# The Harbor OpenHands agent forwards OPENHANDS_* env vars (stripping prefix).
export OPENHANDS_ENABLE_DEFAULT_CONDENSER=false
# Disable history truncation to prevent infinite condensation loops when context
# is exceeded. With this off, ContextWindowExceededError is raised cleanly instead
# of looping through condenser requests that can never reduce essential events.
export OPENHANDS_AGENT_ENABLE_HISTORY_TRUNCATION=false

#-----------------------
# vLLM endpoint for Docker containers
#-----------------------
# OpenHands runs inside Docker containers (not on the host). The containers reach
# the host's vLLM server via the Docker bridge gateway (172.17.0.1 on Linux).
# Override VLLM_API_BASE if your Docker bridge uses a different gateway IP.
VLLM_PORT=8000
VLLM_API_BASE="${VLLM_API_BASE:-http://172.17.0.1:${VLLM_PORT}/v1}"
echo "vLLM API base for Docker containers: $VLLM_API_BASE"

#-----------------------
# Dataset setup
#-----------------------
# Prepare datasets first (downloads from HuggingFace and extracts tasks):
# uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests
# uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/OpenThoughts-TB-dev
DATA_DIR="$HOME/data/harbor"
TRAIN_DATA="['$DATA_DIR/CodeContests']"
EVAL_DATA="['$DATA_DIR/OpenThoughts-TB-dev']"

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="codecontest-openhands"
TRIALS_DIR="$HOME/$RUN_NAME/trials_run"
CKPTS_DIR="$HOME/$RUN_NAME/ckpts"
EXPORTS_DIR="$HOME/$RUN_NAME/exports"
# Logs (trainer + tee) go under my_logs/ in the repo root when run from SkyRL-main.
LOG_DIR="my_logs/$RUN_NAME"
mkdir -p "$LOG_DIR"
# To save the full run log when you interrupt: ... 2>&1 | stdbuf -oL tee "$LOG_DIR/training.log"

#-----------------------
# Training setup
#-----------------------
MINI_BATCH_SIZE=2
MAX_MODEL_LEN=16384
APPLY_OVERLONG_FILTERING=true

# Dr. GRPO parameters
LOSS_REDUCTION="seq_mean_token_sum_norm"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false

CHAT_TEMPLATE_PATH="$(dirname "$0")/../../../skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"

#----------------
# Infrastructure setup
#----------------
NUM_GPUS=1
ENABLE_RATE_LIMITING=true
# OpenHands trials are heavier than terminus-2 but Docker runs locally.
TRAJECTORIES_PER_SECOND=2
MAX_CONCURRENCY=4

# Run SkyRL command with OpenHands agent
uv run --isolated --extra fsdp --extra harbor -m examples.train_integrations.harbor.entrypoints.main_harbor \
  data.train_data=$TRAIN_DATA \
  data.val_data=$EVAL_DATA \
  trainer.policy.model.path=Qwen/Qwen3-1.7B \
  generator.served_model_name=Qwen3-1.7B \
  hydra.searchpath=['file://examples/train_integrations/harbor'] \
  +harbor_trial_config=openhands \
  ++harbor_trial_config.trials_dir=$TRIALS_DIR \
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
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  +generator.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  +generator.engine_init_kwargs.enable_log_requests=false \
  trainer.epochs=1 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=4 \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.gpu_memory_utilization=0.5 \
  trainer.logger=wandb \
  trainer.project_name=harbor \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.enforce_eager=false \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host=0.0.0.0 \
  generator.http_endpoint_port=8000 \
  ++harbor_trial_config.agent.kwargs.api_base="${VLLM_API_BASE}" \
  +generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  +generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  +generator.rate_limit.max_concurrency=$MAX_CONCURRENCY
