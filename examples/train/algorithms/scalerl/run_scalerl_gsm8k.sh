#!/bin/bash
set -x

# Example ScaleRL-style run on GSM8K using the standard SkyRL trainer.
#
# Run data preparation first:
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/algorithms/scalerl/run_scalerl_gsm8k.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=4
LOGGER="wandb"  # change to "console" to print to stdout

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
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
  trainer.algorithm.policy_loss_type=cispo \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.zero_variance_filter=true \
  trainer.algorithm.loss_reduction=prompt_mean \
  trainer.algorithm.advantage_batch_normalize=true \
  trainer.algorithm.adaptive_prompt_filtering.enabled=true \
  trainer.algorithm.adaptive_prompt_filtering.metric=pass_rate \
  trainer.algorithm.adaptive_prompt_filtering.threshold=0.9 \
  trainer.algorithm.adaptive_prompt_filtering.min_history=5 \
  trainer.algorithm.adaptive_prompt_filtering.warmup_epochs=1 \
  trainer.algorithm.adaptive_prompt_filtering.min_active_ratio=0.1 \
  trainer.policy.model.upcast_logits_to_fp32=true \
  trainer.ref.model.upcast_logits_to_fp32=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  trainer.logger="$LOGGER" \
  trainer.project_name="scalerl_gsm8k" \
  trainer.run_name="scalerl_gsm8k_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/scalerl_gsm8k_1.5B_ckpt" \
  $@
