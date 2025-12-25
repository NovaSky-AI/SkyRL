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

DATA_DIR="$HOME/data/lcb"
train_data="['${DATA_DIR}/deepcoder_train_short.json']"
val_data="['${DATA_DIR}/test_livecodebench_short.json']"

# NOTE (sumanthrh): micro_train_batch_size and micro_forward_batch_size can be tuned
uv run --isolated --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data=$train_data \
  data.val_data=$train_data \
  trainer.policy.model.path=$MODEL_NAME \
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
  trainer.policy_mini_batch_size=4 \
  trainer.train_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=29000 \
  generator.max_input_length=29000 \
  generator.sampling_params.max_generate_length=3000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.ckpt_interval=100000 \
  trainer.flash_attn=false \
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
  trainer.run_name="skyrlcode_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/lcb_3B_ckpt" \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@
