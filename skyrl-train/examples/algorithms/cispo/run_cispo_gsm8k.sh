#!/bin/bash
set -x

# Example of CISPO policy loss training
# Clipped Importance Sampling Weight Policy Optimization (CISPO) for better RL efficiency

# Run data preparation first:
# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/algorithms/cispo/run_cispo_gsm8k.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=4
LOGGER="wandb"  # change to "console" to print to stdout

# Configure CISPO parameters
POLICY_LOSS="cispo"
CISPO_EPS_CLIP_LOW=0
CISPO_EPS_CLIP_HIGH=5
USE_KL_LOSS=false

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.policy_loss_type="$POLICY_LOSS" \
  trainer.algorithm.cispo.cispo_eps_clip_low=$CISPO_EPS_CLIP_LOW \
  trainer.algorithm.cispo.cispo_eps_clip_high=$CISPO_EPS_CLIP_HIGH \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="cispo_gsm8k" \
  trainer.run_name="cispo_gsm8k_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/cispo_gsm8k_1.5B_ckpt" \
  $@
