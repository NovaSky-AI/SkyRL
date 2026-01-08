set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-3B-Instruct on SearchR1 data.
# export WANDB_API_KEY=<your_key_here>
# bash examples/livecodebench/run_lcb.sh

export WANDB_API_KEY="6a25ce5b41815c557d6fe8aecb8bac2dd6b1bea0"


NUM_NODES=1
NUM_GPUS=1
LOGGER="wandb"


MODEL_NAME="Qwen/Qwen3-8B"


FLASH_ATTN=true
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=1

train_data="$HOME/data/lcb/deepcoder_train_short.json"

CKPTS_DIR="$HOME/ckpts_hover"
EXPORTS_DIR="$HOME/hf_ckpts_hover"

CHAT_TEMPLATE_PATH="$HOME/SkyRL/skyrl-train/examples/dspy/qwen3_thinking_acc.jinja2"
# train_data="['${DATA_DIR}/deepcoder_train_short.json']"
# val_data="['${DATA_DIR}/test_livecodebench_short.json']"

# NOTE (sumanthrh): micro_train_batch_size and micro_forward_batch_size can be tuned
uv run --isolated --extra dspy --extra vllm -m examples.dspy.entrypoints.main_dspy \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data=$train_data \
  data.val_data=$train_data \
  +dspy.program="Hover_query_gen" \
  +dspy.benchmark_name="hover" \
  +dspy.local_reward_fn="hover_query_reward_fn" \
  +dspy.final_reward_fn="hover_final_reward_fn" \
  +generator.engine_init_kwargs.custom_chat_template_chat_completion_path=$CHAT_TEMPLATE_PATH \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.policy.model.lora.rank=0 \
  trainer.policy.model.lora.alpha=16 \
  trainer.policy.model.lora.dropout=0 \
  trainer.policy.model.lora.lora_sync_path="$HOME/skyrl_lora_sync" \
  trainer.policy.model.lora.target_modules="all-linear" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=8000 \
  trainer.epochs=20 \
  trainer.policy_mini_batch_size=64 \
  trainer.train_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=29000 \
  generator.max_input_length=29000 \
  generator.sampling_params.max_generate_length=3000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.hf_save_interval=10 \
  trainer.ckpt_interval=10 \
  trainer.flash_attn=$FLASH_ATTN \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=lcb \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.7 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  trainer.logger="wandb" \
  trainer.project_name="skyrl" \
  trainer.run_name="skyrl_hover" \
  trainer.resume_mode=null \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  $@
