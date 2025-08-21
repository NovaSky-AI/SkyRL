set -x

# Test script for Qwen3 thinking tokens

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=1
LOGGER="console"
INFERENCE_BACKEND="vllm"

# Test WITH thinking tokens
echo "=== TESTING WITH THINKING TOKENS ==="
uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-0.6B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=4 \
  trainer.eval_before_train=true \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.critic_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=1 \
  generator.gpu_memory_utilization=0.8 \
  generator.enable_thinking_tokens=true \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_thinking_test" \
  trainer.run_name="qwen3_with_thinking" \
  trainer.resume_mode=null \
  trainer.dump_data_batch=true \
  trainer.ckpt_path="$HOME/ckpts/qwen3_thinking_test" \
  $@

echo "=== TESTING WITHOUT THINKING TOKENS ==="
uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-0.6B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=4 \
  trainer.eval_before_train=true \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.critic_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=1 \
  generator.gpu_memory_utilization=0.8 \
  generator.enable_thinking_tokens=false \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_thinking_test" \
  trainer.run_name="qwen3_without_thinking" \
  trainer.resume_mode=null \
  trainer.dump_data_batch=true \
  trainer.ckpt_path="$HOME/ckpts/qwen3_no_thinking_test" \
  $@

# Add these lines to your script:
trainer.train_batch_size=1 \          # Process only 1 sample
trainer.eval_batch_size=1 \           # Eval only 1 sample  
trainer.epochs=1 \                    # Only 1 epoch
trainer.max_steps=1 \                 # Stop after 1 step
trainer.eval_interval=-1 \            # Disable evaluation
trainer.eval_before_train=false \     # Skip initial eval