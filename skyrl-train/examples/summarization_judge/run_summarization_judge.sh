#!/bin/bash
set -x

# Colocated GRPO LoRA training + generation for summarization with LLM as Judge.
#
# 1. Prepare dataset:
#    uv run examples/summarization_judge/summarization_dataset.py \
#        --input_file /path/to/your/data.jsonl \
#        --output_dir $HOME/data/summarization_judge
#
# 2. Set API key in .env.summarization_judge:
#    OPENAI_API_KEY=sk-...
#
# 3. Run training:
#    bash examples/summarization_judge/run_summarization_judge.sh

DATA_DIR="$HOME/data/summarization_judge"
CKPT_PATH="$HOME/ckpts/summarization_judge"

NUM_NODES=2
NUM_GPUS=4  # per node
TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))  # 8 total
LOGGER="console"  # change to "wandb" for W&B logging

INFERENCE_BACKEND="vllm"


uv run --isolated --extra $INFERENCE_BACKEND --env-file .env.summarization_judge -m examples.summarization_judge.main_summarization_judge \
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
  trainer.epochs=4 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=2048 \
  trainer.policy.optimizer_config.lr=3.0e-5 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=summarization_judge \
  environment.skyrl_gym.summarization_judge.model="gpt-4o-mini" \
  environment.skyrl_gym.summarization_judge.temperature=0.0 \
  generator.n_samples_per_prompt=2 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="summarization_judge" \
  trainer.run_name="summarization_judge_grpo_lora" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  $@
