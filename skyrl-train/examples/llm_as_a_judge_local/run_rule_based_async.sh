#!/bin/bash
set -x

# ============================================================================
# Rule-based GRPO training on GSM8K (asynchronous)
# ============================================================================
#
# Async companion to run_rule_based.sh — generation runs one step ahead of
# training for improved throughput.  Uses disaggregated GPUs (async requires
# colocate_all=false).
#
# Prerequisites:
#   1. Prepare the dataset:
#      uv run examples/gsm8k/gsm8k_dataset.py \
#          --output_dir $HOME/data/gsm8k
#
#   2. Run training:
#      bash examples/llm_as_a_judge_local/run_rule_based_async.sh
#
# GPU layout (2 GPUs, disaggregated — async requires colocate_all=false):
#   - GPU 1: FSDP policy model (training)
#   - GPU 2: vLLM inference engine (generation, runs 1 step ahead)
#   - Reward: instant string matching (zero GPU cost)
#
# Works on any GPU with ≥16 GB VRAM and bfloat16 support (e.g. L4, A10G, A100).
# ============================================================================

DATA_DIR="$HOME/data/gsm8k"
CKPT_PATH="$HOME/ckpts/async_gsm8k_rule_based"

NUM_POLICY_GPUS=1
NUM_INFERENCE_ENGINES=1
TP_SIZE=1
LOGGER=wandb

uv run --isolated --extra vllm -m examples.async.main_async \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=false \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.critic_num_gpus_per_node=0 \
  trainer.placement.ref_num_gpus_per_node=0 \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=2 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k-async" \
  trainer.run_name="gsm8k_async_rule_based" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  environment.env_class=gsm8k \
  "$@"
