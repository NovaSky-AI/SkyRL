set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.

DATA_DIR="data/gsm8k"

uv run --isolated --extra vllm --env-file .env -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="gae" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  trainer.placement.critic_num_gpus_per_node=8 \
  trainer.critic.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  generator.num_inference_engines=8 \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_before_train=true \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.critic_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=32 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.eval_interval=1 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=512 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.critic.optimizer_config.lr=2.0e-5 \
  trainer.algorithm.value_clip=0.5 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.advantage_batch_normalize=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.ppo_loss_type="dual_clip" \
  trainer.algorithm.normalize_reward=true \
  trainer.gradient_checkpointing=false \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.eval_n_samples_per_prompt=1 \
  generator.eval_sampling_params.temperature=0.0 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="wandb" \
  trainer.project_name="verl_skyrl_gsm8k" \
  trainer.run_name="skyrl_qwen1.5b_llm_ppo_fsdp_0.8.5_no_adv_normalize_eval_fix" \
  trainer.resume_mode=null \
  trainer.ckpt_path="/mnt/local_storage/gsm8k_1.5B_ckpt"