set -ex

# wandb api key.
# export WANDB_API_KEY=YOUR_KEY_HERE

# Pick the sandbox provider and provide the credentials.
# export DAYTONA_API_KEY=YOUR_KEY_HERE
# export MODAL_TOKEN_ID=YOUR_KEY_HERE
# export MODAL_TOKEN_SECRET=YOUR_KEY_HERE

# Prepare dataset first (downloads from HuggingFace and extracts tasks):
# python examples/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests

DATA_DIR="$HOME/data/harbor"
TRAIN_DATA="['$DATA_DIR/CodeContests']"

CHAT_TEMPLATE_PATH="$(dirname "$0")/../../skyrl_train/utils/templates/qwen3_acc_thinking.jinja2"
TRIALS_DIR="$HOME/trials_run"

#----------------
# Infrastructure setup
#----------------
NUM_GPUS=4
ENABLE_RATE_LIMITING=true  # Enable rate/concurrency limiting for trajectory submissions
TRAJECTORIES_PER_SECOND=5  # Maximum trajectories per second (must be >= 1.0, fractional values like 1.5 are supported). null or omit to disable rate limiting
MAX_CONCURRENCY=512        # Maximum concurrent trial.run() calls allowed (must be >= 1). null or omit to disable concurrency limiting


uv run --isolated --extra vllm --extra harbor -m examples.harbor.entrypoints.main_harbor_generate \
  data.train_data=$TRAIN_DATA \
  hydra.searchpath=['file://examples/harbor'] \
  +harbor_trial_config=default \
  ++harbor_trial_config.trials_dir=$TRIALS_DIR \
  ++harbor_trial_config.trial_name="dummy" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  generator.served_model_name="Qwen2.5-1.5B-Instruct" \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=8000 \
  generator.sampling_params.max_generate_length=4096 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.gpu_memory_utilization=0.8 \
  +generator.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.train_batch_size=$NUM_GPUS \
  trainer.policy_mini_batch_size=$NUM_GPUS \
  trainer.logger=console \
  +generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  +generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  +generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  $@
