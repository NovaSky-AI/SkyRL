set -x

# Colocated DAPO with blockwise FP8 training + FP8 rollout for Qwen3.5-35B-A3B-Base (MoE).
# Hardware: 2 nodes of 8xH100
#
# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/fp8/run_fp8_hopper_blockwise_qwen35_35b_a3b.sh
#
# FP8 here covers trainer linear-layer GEMMs plus rollout weights via
# representation-preserving weight sync (serialized_weight_sync_mode=blockwise):
# vLLM receives the trainer-produced FP8 payloads and block scales instead of
# re-quantizing a BF16 export.

MODEL_NAME="Qwen/Qwen3.5-35B-A3B-Base"
DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER="wandb"  # change to "console" to print to stdout

# Colocated by default: training and inference share the same GPUs. For a
# disaggregated (non-colocated) run, set COLOCATE_ALL=false and split the GPUs,
# e.g. trainer.placement.policy_num_gpus_per_node=4 with the remaining GPUs
# given to the inference engines via generator.inference_engine.num_engines.
COLOCATE_ALL=${COLOCATE_ALL:-true}

NUM_NODES=2
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8

MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

# ---- FP8: trainer GEMMs + rollout weight sync ----
# Hopper blockwise FP8 uses exact FP32 block scales end to end. Both exported
# variables are defaulted and validated by SkyRL at startup; set for clarity.
MEGATRON_FP8=e4m3
MEGATRON_FP8_RECIPE=blockwise
MEGATRON_FP8_AMAX_COMPUTE_ALGO=most_recent
MEGATRON_TP_ONLY_AMAX_RED=false
SERIALIZED_WEIGHT_SYNC_MODE=blockwise
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export VLLM_USE_DEEP_GEMM_E8M0=0

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="regular" \
  trainer.algorithm.overlong_buffer_len=4096 \
  trainer.algorithm.overlong_buffer_penalty_factor=1.0 \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.clip_ratio_c=10.0 \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  generator.apply_overlong_filtering=true \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.max_generate_length=8192 \
  generator.sampling_params.logprobs=1 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.max_generate_length=8192 \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=$COLOCATE_ALL \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.fp8=$MEGATRON_FP8 \
  trainer.ref.megatron_config.fp8=$MEGATRON_FP8 \
  trainer.policy.megatron_config.fp8_recipe=$MEGATRON_FP8_RECIPE \
  trainer.ref.megatron_config.fp8_recipe=$MEGATRON_FP8_RECIPE \
  trainer.policy.megatron_config.fp8_amax_compute_algo=$MEGATRON_FP8_AMAX_COMPUTE_ALGO \
  trainer.ref.megatron_config.fp8_amax_compute_algo=$MEGATRON_FP8_AMAX_COMPUTE_ALGO \
  trainer.policy.megatron_config.transformer_config_kwargs.tp_only_amax_red=$MEGATRON_TP_ONLY_AMAX_RED \
  trainer.ref.megatron_config.transformer_config_kwargs.tp_only_amax_red=$MEGATRON_TP_ONLY_AMAX_RED \
  generator.inference_engine.serialized_weight_sync_mode=$SERIALIZED_WEIGHT_SYNC_MODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=16 \
  trainer.epochs=20 \
  trainer.max_training_steps=400 \
  trainer.eval_batch_size=512 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=2048 \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyrl_fp8" \
  trainer.run_name="fp8_hopper_blockwise_qwen35_35b_a3b" \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=null \
  trainer.max_ckpts_to_keep=3 \
  $@
