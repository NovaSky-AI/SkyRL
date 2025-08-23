set -x

# Test script for standardized chat template system

DATA_DIR="$HOME/data/gsm8k_mini"
NUM_GPUS=1
LOGGER="wandb"
INFERENCE_BACKEND="vllm"

COMMON_ARGS="
  data.train_data=['$DATA_DIR/train.parquet'] \
  data.val_data=['$DATA_DIR/validation.parquet'] \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path=Qwen/Qwen3-0.6B \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1 \
  trainer.policy_mini_batch_size=1 \
  trainer.critic_mini_batch_size=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
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
  trainer.logger=$LOGGER \
  trainer.project_name=updated_template_system \
  trainer.resume_mode=null \
  trainer.dump_data_batch=true"

# qwen3 with thinking tokens (using boolean and model name)
echo "=== TEST 1: QWEN3 WITH THINKING TOKENS (BOOLEAN) ==="
uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  $COMMON_ARGS \
  generator.enable_thinking_tokens=true \
  trainer.run_name="test1_qwen3_with_thinking" \
  trainer.ckpt_path="$HOME/ckpts/test1_qwen3_with_thinking" \
  $@

# qwen3 without thinking tokens (using boolean and model name)
echo "=== TEST 2: QWEN3 WITHOUT THINKING TOKENS (BOOLEAN) ==="
uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  $COMMON_ARGS \
  generator.enable_thinking_tokens=false \
  trainer.run_name="test2_qwen3_without_thinking" \
  trainer.ckpt_path="$HOME/ckpts/test2_qwen3_without_thinking" \
  $@

# using named template from CUSTOM_CHAT_TEMPLATES
echo "=== TEST 3: NAMED TEMPLATE (custom_thinking_remover) ==="
uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  $COMMON_ARGS \
  generator.chat_template.source="name" \
  generator.chat_template.name_or_path="custom_thinking_remover" \
  trainer.run_name="test3_named_template" \
  trainer.ckpt_path="$HOME/ckpts/test3_named_template" \
  $@

# using a simple file template 
echo "=== TEST 4: FILE TEMPLATE (custom_chat_template.j2) ==="
uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  $COMMON_ARGS \
  generator.chat_template.source="file" \
  generator.chat_template.name_or_path="/home/ubuntu/SkyRL/skyagent/skyagent/functional/templates/custom_chat_template.j2" \
  trainer.run_name="test4_file_template_simple" \
  trainer.ckpt_path="$HOME/ckpts/test4_file_template_simple" \
  $@

# using file template for qwen3 format with thinking support
echo "=== TEST 5: FILE TEMPLATE (qwen3_acc_thinking.jinja2) ==="
uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  $COMMON_ARGS \
  generator.chat_template.source="file" \
  generator.chat_template.name_or_path="/home/ubuntu/SkyRL/skyagent/skyagent/functional/templates/qwen3_acc_thinking.jinja2" \
  trainer.run_name="test5_file_template_qwen3" \
  trainer.ckpt_path="$HOME/ckpts/test5_file_template_qwen3" \
  $@

# using named template explicitly instead of boolean (compare diff from test 1)
echo "=== TEST 6: NAMED TEMPLATE (qwen3_with_thinking via name) ==="
uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  $COMMON_ARGS \
  generator.chat_template.source="name" \
  generator.chat_template.name_or_path="qwen3_with_thinking" \
  trainer.run_name="test6_named_qwen3_thinking" \
  trainer.ckpt_path="$HOME/ckpts/test6_named_qwen3_thinking" \
  $@

echo "=== ALL TESTS COMPLETE ==="

