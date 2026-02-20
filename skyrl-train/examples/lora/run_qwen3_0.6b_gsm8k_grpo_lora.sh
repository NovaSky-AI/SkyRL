set -x

set -x

# Colocated GRPO LoRA training + generation for Qwen3-0.6B on GSM8K with LLM as Judge.

# 1. Prepare dataset:
# uv run examples/llm_as_a_judge/gsm8k_dataset_judge.py --output_dir $HOME/data/gsm8k_llm_judge
# 2. Set API key in .env.llm_judge:
# OPENAI_API_KEY=sk-...
# 3. Run training:
# bash examples/lora/run_qwen3_0.6b_gsm8k_grpo_lora.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned

DATA_DIR="/mnt/workspace/datasets/gsm8k_with_reward"
NUM_NODES=2
NUM_GPUS=4  # per node
TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))  # 8 total
LOGGER="console"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"


uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-0.6B" \
  trainer.placement.colocate_all=true \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$TOTAL_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=512 \
  trainer.policy.optimizer_config.lr=3.0e-5 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=llm_as_a_judge \
  environment.skyrl_gym.llm_as_a_judge.model="gpt-4o-mini" \
  generator.n_samples_per_prompt=2 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_qwen3_0.6b_lora" \
  trainer.run_name="gsm8k_qwen3_0.6b_lora_grpo" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_qwen3_0.6b_lora_ckpt" \
  $@
