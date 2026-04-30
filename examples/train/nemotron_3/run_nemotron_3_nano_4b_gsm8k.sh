set -x

# Colocated GRPO training+generation for NVIDIA-Nemotron-3-Nano-4B-BF16 on GSM8K with Megatron.
#
# Prerequisites:
#   See examples/train/nemotron_3/README.md for required dependency changes.
#
# Setup:
# 1. Prepare GSM8K data:
#    uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#
# 2. Run training:
#    bash examples/train/nemotron_3/run_nemotron_3_nano_4b_gsm8k.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=8
LOGGER="wandb"  # change to "console" to print to stdout
MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

INFERENCE_BACKEND="vllm"

# Megatron parallelism: TP=4, PP=1 uses 4 GPUs for training, leaving 4 for inference
MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1

# Inference: 8 engines with TP=1 each (must match NUM_GPUS when colocate_all=true)
NUM_INFERENCE_ENGINES=8
INFERENCE_TP=1

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_TP \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.use_sample_packing=true \
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
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_megatron_nemotron" \
  trainer.run_name="gsm8k_megatron_nemotron_3_nano_4b_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_megatron_nemotron_3_nano_ckpt" \
  $@
