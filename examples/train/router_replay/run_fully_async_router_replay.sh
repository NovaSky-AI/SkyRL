set -x

# Fully async GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.
# This bash script is copied from examples/async/async_run_gsm8k.sh, except for:
# - running examples.train.fully_async.main_fully_async
# - setting the generator.batched=false.
# - colocate_all=false
# - the various generator configs at the end (http, chat template, etc.)

# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/router_replay/run_fully_async_router_replay.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned

# You can override the default values with e.g.: `NUM_GPUS=1 bash examples/train/fully_async/fully_async_run_gsm8k.sh`.

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_INFERENCE_GPUS:=4}"
: "${NUM_POLICY_GPUS:=4}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout / or use wandb

: "${INFERENCE_BACKEND:=vllm}"

# Fully async specific configuration knobs:
: "${MINI_BATCH_SIZE:=64}"
: "${MAX_STALENESS_STEPS:=4}"
: "${NUM_PARALLEL_GENERATION_WORKERS:=$(( MINI_BATCH_SIZE * (MAX_STALENESS_STEPS + 1) ))}"

TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

# moonlight16b
MODEL_NAME="moonshotai/Moonlight-16B-A3B-Instruct"

MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=4
MEGATRON_ETP=1

# optimizer offload
OPTIMIZER_OFFLOAD=True
OPTIMIZER_OFFLOAD_FRACTION=1.0

NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=4

# router replay (r3)
ROUTER_REPLAY=True
DISTRIBUTED_EXECUTION_BACKEND="mp"

RUN_NAME=fully_async_moonlight_using_r3

FLASH_ATTN=false

uv run --isolated --extra megatron -m examples.train.fully_async.main_fully_async \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.fully_async.max_staleness_steps=${MAX_STALENESS_STEPS} \
  trainer.fully_async.num_parallel_generation_workers=${NUM_PARALLEL_GENERATION_WORKERS} \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=false \
  trainer.strategy=megatron \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_torch_optimizer_for_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.distributed_executor_backend=$DISTRIBUTED_EXECUTION_BACKEND \
  generator.inference_engine.enable_return_routed_experts=$ROUTER_REPLAY \
  trainer.policy.megatron_config.moe_enable_routing_replay=$ROUTER_REPLAY \
  generator.inference_engine.tensor_parallel_size=$NUM_INFERENCE_GPUS \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=4 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${MINI_BATCH_SIZE} \
  trainer.policy_mini_batch_size=${MINI_BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k-async" \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  generator.inference_engine.enforce_eager=true \
  trainer.flash_attn=$FLASH_ATTN \
  $@