set -ex

# wandb api key.
# export WANDB_API_KEY=YOUR_KEY_HERE

# Pick the sandbox provider and provide the credentials.
# export DAYTONA_API_KEY=YOUR_KEY_HERE
# export MODAL_TOKEN_ID=YOUR_KEY_HERE
# export MODAL_TOKEN_SECRET=YOUR_KEY_HERE

# Prepare datasets first (downloads from HuggingFace and extracts tasks):
# python examples/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests
# python examples/harbor/prepare_harbor_dataset.py --dataset open-thoughts/OpenThoughts-TB-dev

DATA_DIR="$HOME/data/harbor"
TRAIN_DATA="['$DATA_DIR/CodeContests']"
EVAL_DATA="['$DATA_DIR/OpenThoughts-TB-dev']"

CHAT_TEMPLATE_PATH="$(dirname "$0")/../../skyrl_train/utils/templates/qwen3_acc_thinking.jinja2"

RUN_NAME="codecontest"
TRIALS_DIR="$HOME/$RUN_NAME/trials_run"
CKPTS_DIR="$HOME/$RUN_NAME/ckpts"
EXPORTS_DIR="$HOME/$RUN_NAME/exports"
LOG_DIR="/tmp/skyrl-logs/$RUN_NAME"

NUM_GPUS=4
MINI_BATCH_SIZE=32
MAX_MODEL_LEN=32768

# Run SkyRL command
uv run --isolated --extra vllm --extra harbor -m examples.harbor.entrypoints.main_harbor \
  data.train_data=$TRAIN_DATA \
  data.val_data=$EVAL_DATA \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.served_model_name=Qwen3-8B \
  hydra.searchpath=['file://examples/harbor'] \
  +harbor_trial_config=default \
  ++harbor_trial_config.trials_dir=$TRIALS_DIR \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.algorithm.advantage_estimator=grpo \
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
  trainer.epochs=3 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=true \
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
  trainer.algorithm.use_kl_loss=true \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.8 \
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
  generator.http_endpoint_host=127.0.0.1 \
  generator.http_endpoint_port=8000
