set -x

# Colocated DAPO training+generation for Qwen2.5-32B on the original DAPO dataset with Int8 rollouts.
# The configuration is tested on 2 8xH100 GPUs.

# DATA_DIR=$HOME/data/dapo bash examples/algorithms/dapo/prepare_dapo_data.sh
# export WANDB_API_KEY=<your_key_here>
# bash examples/flash_rl/run_dapo_repro_flashrl_32b_int8.sh

DATA_DIR="$HOME/data/dapo"
NUM_GPUS=16
LOGGER="wandb"  # change to "console" to print to stdout

# main DAPO parameters
EPS_CLIP_LOW=0.2
EPS_CLIP_HIGH=0.28
DYNAMIC_SAMPLING_TYPE=filter
DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES=30
LOSS_REDUCTION="token_mean"
# applies overlong filtering (but not soft overlong punishment)
APPLY_OVERLONG_FILTERING=true
# apply soft overlong punishment with custom trainer impl in main_dapo_flashrl.py
OVERLONG_BUFFER_LEN=4096
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# other DAPO parameters
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_RESPONSE_LENGTH=20480
MAX_PROMPT_LENGTH=2048

TIS_IMP_RATIO_CAP=8.0
USE_TIS=true
LOGPROBS=0

CKPT_PATH="$HOME/ckpts/dapo_32b_ckpt"

uv run --isolated --extra flashrl --env-file examples/flash_rl/.env.int8 -m examples.flash_rl.main_dapo_flashrl \
  data.train_data="['$DATA_DIR/dapo-math-17k-cleaned.parquet']" \
  data.val_data="['$DATA_DIR/aime-2024-cleaned.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  +trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
  +trainer.algorithm.overlong_buffer.penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW \
  trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH \
  trainer.algorithm.dynamic_sampling.type=$DYNAMIC_SAMPLING_TYPE \
  trainer.algorithm.dynamic_sampling.max_sample_batches=$DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.logprobs=$LOGPROBS \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.algorithm.use_tis=$USE_TIS \
  trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
  trainer.policy.model.path="Qwen/Qwen2.5-32B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=8 \
  generator.inference_engine_tensor_parallel_size=2 \
  trainer.epochs=20 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.backend=vllm \
  generator.weight_transfer_threshold_cuda_ipc_GB=4.0 \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=16 \
  generator.gpu_memory_utilization=0.6 \
  trainer.logger="$LOGGER" \
  trainer.project_name="dapo_repro" \
  trainer.run_name="dapo_repro_32b_int8" \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.enforce_eager=false \
  $@
