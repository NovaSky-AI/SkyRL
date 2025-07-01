set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.

# export WANDB_API_KEY=<your_key_here>
# bash examples/gsm8k/run_gsm8k.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned
# TODO (sumanthrh): Remove the `resume_mode` and `ckpt_path` arguments before release.

DATA_DIR="data/skyrl_v0_293"

MAX_INPUT_LENGTH=31232
MAX_RESPONSE_LENGTH=1536
MAX_ITERATIONS=1
NUM_GPUS=1
MAX_PARALLEL_AGENTS=4

uv run --isolated --extra vllm --extra swebench --env-file .env -m skyrl_train.entrypoints.main_agent \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_interval=-1 \
  trainer.eval_before_train=false \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=8 \
  trainer.critic_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=$MAX_INPUT_LENGTH \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.max_turns=$MAX_ITERATIONS \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.temperature=0.5 \
  generator.sampling_params.top_p=0.95 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=null \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.8 \
  swebench_config.max_parallel_agents=$MAX_PARALLEL_AGENTS \
  swebench_config.max_eval_parallel_agents=$MAX_PARALLEL_AGENTS \
  swebench_config.log_messages_dir="/home/ubuntu/tgriggs/skyrl_logs/swebench_test" \
  swebench_config.remove_think_tokens=false \
  swebench_config.qwen3_enable_thinking=false \
  trainer.logger="wandb" \
  trainer.project_name="sumanth-swebench-skyrl" \
  trainer.run_name="swebench_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="/home/ubuntu/tgriggs/ckpts/gsm8k_1.5B_ckpt" 