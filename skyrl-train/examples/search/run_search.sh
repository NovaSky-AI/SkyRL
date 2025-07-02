set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-3B-Instruct on SearchR1 data.
# export WANDB_API_KEY=<your_key_here>
# bash examples/search/run_search.sh

DATA_DIR="$HOME/data/searchR1"

# Configuration sections:
# - Dataset: train/val data paths
# - Algorithm: GRPO settings, learning rate, KL loss
# - Model: model path, placement, FSDP settings
# - Training: epochs, batch sizes, gradient settings
# - Length limits: prompt and generation lengths
# - Generator: VLLM backend, colocation, GPU settings  
# - Multi-turn: async, batching, sampling settings
# - Logging: wandb project and run name
# - Checkpointing: intervals, paths, resume settings
# - Evaluation: batch size, intervals, timing

uv run --isolated --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="Qwen/Qwen2.5-3B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=2 \
  trainer.epochs=1 \
  trainer.policy.optimizer_config.num_warmup_steps=94 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=1000 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.gpu_memory_utilization=0.6 \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=5 \
  generator.max_turns=2 \
  generator.use_conversation_multi_turn=false \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  environment.env_class="search" \
  generator.max_env_workers=16 \
  trainer.logger="wandb" \
  trainer.project_name="skyrl-search" \
  trainer.run_name="skyrlsearch_env_workers16" \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/searchR1_3B_ckpt_2" \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  $@
  