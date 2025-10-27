set -x

# Script to simulate full context training for GSM8K with Qwen2.5-1.5B-Instruct on 4 GPUs

# NOTE: Make sure to tune the configurations for the setup you wish to test.

DATA_DIR="/mnt/cluster_storage/gsm8k"

export SKYRL_LD_LIBRARY_PATH_EXPORT=true
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH

uv run --isolated --extra vllm -m scripts.full_context.main_full_ctx \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-32B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=2 \
  trainer.placement.policy_num_gpus_per_node=8 \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=4 \
  trainer.epochs=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.critic_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=20480 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="wandb" \
  trainer.project_name="gsm8k_full_ctx" \
  trainer.run_name="gsm8k_full_ctx_test" \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  +trainer.num_dummy_steps=5