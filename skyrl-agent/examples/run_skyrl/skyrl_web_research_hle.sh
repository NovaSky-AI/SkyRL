# Colocated GRPO training+generation for Qwen3-8B on deep search.

set -x

# Basic environment setup
export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
# increase ulimit for ray if applicable
# ulimit -n 65535

DATA_DIR=$HOME/data/web_research
CKPT_DIR=$HOME/ckpts
# save eval results and dump training batch
EXPORT_DIR=$HOME/exports

TRAIN_DATA="$DATA_DIR/textbook_stem_5k_with_instance.parquet"
VAL_DATA="$DATA_DIR/hle_webthinker_converted.parquet"

# Set a writable, shared web cache directory for the web_browser tool
# change if needed
export SKYAGENT_WEB_CACHE_DIR="$DATA_DIR/skyagent_web_cache"
mkdir -p "$SKYAGENT_WEB_CACHE_DIR"

NUM_GPUS=4
LOGGER="wandb"  # change to "console" to print to stdout
seed=1 # for exact reproducibility with verl

INFERENCE_BACKEND="vllm"

# Load environment variables from .env file
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Set a writable, shared web cache directory for the web_browser tool
mkdir -p "${SKYAGENT_WEB_CACHE_DIR:-$DATA_DIR/web_cache}" 

uv run --isolated --frozen --env-file .env --extra skyrl-train -m skyrl_agent.integrations.skyrl_train.skyrl_train_main  \
  data.train_data="['$TRAIN_DATA']" \
  data.val_data="['$VAL_DATA']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.task="./examples/run_skyrl/skyrl_web_research_hle.yaml" \
  trainer.epochs=10 \
  trainer.seed=$seed \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=true \
  trainer.eval_interval=50 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=31232 \
  generator.enforce_eager=true \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.policy.optimizer_config.lr=5e-6 \
  trainer.policy.sequence_parallel_size=2 \
  trainer.ref.sequence_parallel_size=2 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.eps_clip_low=0.4 \
  trainer.algorithm.eps_clip_high=0.4 \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=null \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=1 \
  generator.gpu_memory_utilization=0.75 \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyagent_websearch" \
  trainer.run_name="train-64n8-withoss20b-skyrl-train" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.export_path="$EXPORT_DIR" \
  trainer.dump_data_batch=true \
  $@