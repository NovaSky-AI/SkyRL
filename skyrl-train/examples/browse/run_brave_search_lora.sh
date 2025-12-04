set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-3B-Instruct on SearchR1 data.
# follow the instructions in examples/search/README.md for setting up the dataset
# and for starting the local search server
# export WANDB_API_KEY=<your_key_here>
# bash examples/search/run_search.sh

# path for dataset (.parquet files) containing the prompts and metadata for each question
export HYDRA_FULL_ERROR=1
DATA_DIR="/workspaces/nearaiml/data/searchR1"

uv run --with /workspaces/nearaiml --isolated --frozen --extra vllm -m examples.browse.browse_entrypoint \
  data.train_data="['${DATA_DIR}/train_brave_search.parquet']" \
  data.val_data="['${DATA_DIR}/validation_brave_search.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=94 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  trainer.policy.model.lora.lora_sync_path="/tmp/skyrl_lora_sync" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=2 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.gpu_memory_utilization=0.5 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=8192 \
  generator.sampling_params.max_generate_length=1536 \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=5 \
  generator.max_turns=6 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</search>", "</answer>"]' \
  environment.env_class="browse" \
  environment.skyrl_gym.max_env_workers=8 \
  environment.skyrl_gym.browse.tool_call_parser="qwen3" \
  trainer.logger="wandb" \
  trainer.project_name="skyrl-browse" \
  trainer.run_name="skyrl-browse_6turns_maxgeneratelen_1536_Qwen3-8B_lora" \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/skyrl-browse_6turns_maxgeneratelen_1536_Qwen3-8B_lora" \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  trainer.export_path="$HOME/skyrl-search_6turns_maxgeneratelen_1536_Qwen3-8B_lora/exports" \
  trainer.eval_interval=50 \
  $@
  