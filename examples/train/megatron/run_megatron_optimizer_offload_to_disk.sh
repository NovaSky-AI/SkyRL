set -x

# Colocated GRPO for Qwen3-0.6B on GSM8K with Megatron, with the optimizer state (fp32 master weights and Adam
# moments) kept on node-local disk and streamed through the GPU during the optimizer step
# (trainer.policy.megatron_config.optimizer_offload_to_disk). A small model to try the feature on; it pays off
# for models whose optimizer state does not fit on the GPU or in host RAM.

# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/megatron/run_megatron_optimizer_offload_to_disk.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=4
LOGGER="wandb"  # change to "console" to print to stdout
MODEL_NAME="Qwen/Qwen3-0.6B"

# Node-local NVMe with room for 12 B x (params / DP size) per GPU; tmpfs is refused.
OPTIMIZER_STATE_DIR="/mnt/local_storage/skyrl_optimizer_state"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.policy.megatron_config.tensor_model_parallel_size=1 \
  trainer.policy.megatron_config.pipeline_model_parallel_size=2 \
  trainer.policy.megatron_config.optimizer_offload_to_disk.dir="$OPTIMIZER_STATE_DIR" \
  trainer.remove_microbatch_padding=true \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_megatron" \
  trainer.run_name="gsm8k_megatron_optimizer_offload_to_disk" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_megatron_optimizer_offload_to_disk_ckpt" \
  $@
