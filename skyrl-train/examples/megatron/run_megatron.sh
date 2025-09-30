set -x

# Colocated GRPO training+generation for Qwen3-0.6B on GSM8K with Megatron.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/megatron/run_megatron.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=4
LOGGER="wandb"  # change to "console" to print to stdout
MODEL_NAME="Qwen/Qwen3-0.6B"

INFERENCE_BACKEND="vllm" # currently only vllm is supported for megatron

MEGATRON_TP=2
MEGATRON_PP=2
MEGATRON_CP=1

# torch profiler config
ENABLE_TORCH_PROFILER=false
RANKS_TO_PROFILE="[0]"
SAVE_PATH="$HOME/megatron_prof/tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_${MODEL_NAME}"

export SKYRL_PYTHONPATH_EXPORT=1
# make sure PYTHONPATH is set to the location of TransformerEngine installation
export PYTHONPATH="$HOME/anaconda3/lib/python3.12/site-packages"

uv run --isolated --extra $INFERENCE_BACKEND --extra mcore -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.policy.megatron_config.torch_profiler_config.enable=$ENABLE_TORCH_PROFILER \
  trainer.policy.megatron_config.torch_profiler_config.ranks=$RANKS_TO_PROFILE \
  trainer.policy.megatron_config.torch_profiler_config.save_path=$SAVE_PATH \
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
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.6 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_megatron" \
  trainer.run_name="gsm8k_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_${MODEL_NAME}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_megatron_ckpt" \
  $@