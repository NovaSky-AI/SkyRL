set -x

# Non-colocated LoRA GRPO for Nemotron 3.5 Lightning on one 8-GPU node.

DATA_DIR="$HOME/data/gsm8k"
MODEL_NAME="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.policy.language_model_only=true \
  trainer.placement.colocate_all=false \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.policy.megatron_config.tensor_model_parallel_size=1 \
  trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
  trainer.policy.megatron_config.expert_model_parallel_size=4 \
  trainer.policy.megatron_config.lora_config.merge_lora=false \
  trainer.policy.megatron_config.transformer_config_kwargs='{"mtp_num_layers":0,"mtp_hybrid_override_pattern":null,"mtp_use_repeated_layer":false,"gradient_accumulation_fusion":false,"recompute_granularity":"full","recompute_method":"uniform","recompute_num_layers":4,"recompute_modules":[]}' \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  trainer.policy.model.lora.target_modules=all-linear \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.epochs=20 \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.ckpt_interval=5 \
  trainer.max_prompt_length=512 \
  trainer.policy.optimizer_config.lr=2e-4 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=4 \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  generator.inference_engine.max_num_seqs=64 \
  generator.inference_engine.max_num_batched_tokens=32768 \
  generator.inference_engine.engine_init_kwargs='{"max_model_len":262144,"kv_cache_dtype":"fp8","disable_custom_all_reduce":true,"mamba_backend":"triton","mamba_ssm_cache_dtype":"float16"}' \
  generator.batched=true \
  generator.n_samples_per_prompt=8 \
  generator.sampling_params.max_generate_length=1024 \
  environment.env_class=gsm8k \
  trainer.logger=wandb \
  trainer.project_name=gsm8k_megatron_nemotron \
  trainer.run_name=gsm8k_megatron_nemotron_3p5_lightning_30b \
  trainer.resume_mode=null \
  "$@"
