set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-3B-Instruct on SkyRL-SQL-653 data.
# uses deepspeed and 1 node with 8 GPUs.
# huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset
# export WANDB_API_KEY=<your_key_here>
# bash examples/text_to_sql/run_sql_deepspeed.sh

DATA_DIR="$HOME/data/sql"
DB_PATH="$HOME/data/sql/db_files/data"


uv run --isolated --frozen --extra vllm --extra deepspeed -m skyrl_train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-Coder-7B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=deepspeed \
  deepspeed_config.train.zero_optimization.offload_param.device=cpu \
  deepspeed_config.train.zero_optimization.offload_optimizer.device=cpu \
  deepspeed_config.train.zero_optimization.sub_group_size=1e4 \
  deepspeed_config.train.zero_optimization.reduce_bucket_size=1e4 \
  deepspeed_config.train.zero_optimization.stage3_param_persistence_threshold=1e2 \
  deepspeed_config.train.disable_trace_cache=True \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=2 \
  generator.inference_engine_tensor_parallel_size=4 \
  trainer.train_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=6000 \
  generator.max_input_length=15000 \
  generator.sampling_params.max_generate_length=3000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy_mini_batch_size=256 \
  trainer.algorithm.use_kl_loss=false \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=text2sql \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.7 \
  generator.max_turns=5 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.stop='["</sql>", "</solution>"]' \
  generator.eval_sampling_params.stop='["</sql>", "</solution>"]' \
  environment.skyrl_gym.text2sql.db_path=$DB_PATH \
  trainer.logger="wandb" \
  trainer.project_name="skyrlsql" \
  trainer.run_name="skyrlsql_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/sql_32B_ckpt" \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@
