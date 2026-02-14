set -x

# wandb api key.
# export WANDB_API_KEY=YOUR_KEY_HERE

# Pick the sandbox provider and provide the credentials.
# export DAYTONA_API_KEY=YOUR_KEY_HERE
# export MODAL_TOKEN_ID=YOUR_KEY_HERE
# export MODAL_TOKEN_SECRET=YOUR_KEY_HERE

# Prepare datasets (downloads from HuggingFace and extracts tasks automatically)
# Idempotent - skips if already prepared. Output dir auto-derived from dataset name.
TRAIN_DATASET="open-thoughts/OpenThoughts-Agent-v1-RL"
EVAL_DATASET="open-thoughts/OpenThoughts-TB-dev"
DATA_DIR="$HOME/data/harbor"

python examples/harbor/prepare_harbor_dataset.py --dataset $TRAIN_DATASET
python examples/harbor/prepare_harbor_dataset.py --dataset $EVAL_DATASET

TRAIN_DATA="['$DATA_DIR/${TRAIN_DATASET##*/}']"
EVAL_DATA="['$DATA_DIR/${EVAL_DATASET##*/}']"

CHAT_TEMPLATE_PATH="$(dirname "$0")/../../skyrl_train/utils/templates/qwen3_acc_thinking.jinja2"
TRIALS_DIR="$HOME/trials_run"
CKPTS_DIR="$HOME/otagent/ckpts"
EXPORTS_DIR="$HOME/otagent/exports"

# Run SkyRL command
uv run --isolated --extra vllm --extra harbor -m examples.harbor.entrypoints.main_harbor \
  data.train_data=$TRAIN_DATA \
  data.val_data=$EVAL_DATA \
  trainer.policy.model.path=open-thoughts/OpenThinker-Agent-v1-SFT \
  generator.served_model_name=OpenThinker-Agent-v1-SFT \
  hydra.searchpath=['file://examples/harbor'] \
  +harbor_trial_config=default \
  ++harbor_trial_config.trials_dir=$TRIALS_DIR \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=8 \
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  trainer.epochs=3 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=true \
  trainer.eval_interval=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=30720 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger=wandb \
  trainer.project_name=dc-agent \
  trainer.run_name=otagent-rl \
  trainer.resume_mode=latest \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host=127.0.0.1 \
  generator.http_endpoint_port=8000
