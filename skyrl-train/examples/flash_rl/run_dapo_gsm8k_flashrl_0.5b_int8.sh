set -x

# Colocated DAPO training+generation for Qwen2.5-0.5B-Instruct on GSM8K with Int8 rollouts.
# The configuration is tested on 4 H100 GPUs.

# uv run --isolated examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/flash_rl/run_dapo_gsm8k_flashrl_0.5b_int8.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=4
LOGGER="wandb"  # change to "console" to print to stdout

# main DAPO parameters
EPS_CLIP_LOW=0.2
EPS_CLIP_HIGH=0.28
DYNAMIC_SAMPLING_TYPE=filter
DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES=30
LOSS_REDUCTION="token_mean"
# applies overlong filtering (but not soft overlong punishment)
APPLY_OVERLONG_FILTERING=true
# apply soft overlong punishment with custom trainer impl
OVERLONG_BUFFER_LEN=512
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# other DAPO parameters
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_RESPONSE_LENGTH=1024

CKPT_PATH="$HOME/ckpts/gsm8k_0.5B_ckpt"

uv run --isolated --extra flashrl --env-file examples/flash_rl/.env.int8 -- python -m examples.flash_rl.main_dapo_flashrl \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  +trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
  +trainer.algorithm.overlong_buffer.penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW \
  trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.logprobs=0 \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  trainer.algorithm.dynamic_sampling.type=$DYNAMIC_SAMPLING_TYPE \
  trainer.algorithm.dynamic_sampling.max_sample_batches=$DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.algorithm.use_tis=true \
  trainer.algorithm.tis_imp_ratio_cap=2.0 \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
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
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.6 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_flashrl" \
  trainer.run_name="gsm8k_dapo_flashrl_0.5B_int8" \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.enforce_eager=false \
  $@
