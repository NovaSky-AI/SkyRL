set -x

# Colocated DAPO with mixed-precision blockwise FP8 training + FP8 rollout for Qwen3.5-9B-Base:
# everything runs as in run_fp8_hopper_blockwise_qwen35_9b.sh, except that chosen modules stay
# in BF16 on both the trainer and the rollout.
# Hardware: 1 node of 8xH100
#
# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/fp8/run_fp8_hopper_blockwise_mixed_qwen35_9b.sh
#
# Kept in BF16 here: layers 0, 1 and 31, and the q/k/v/o projections of every full-attention
# layer (3, 7, ..., 31). Qwen3.5-9B has 32 layers; every 4th is full attention and the rest are
# Gated DeltaNet (GDN) linear attention, whose projections have different names on both sides,
# so the attention patterns below match only the full-attention layers.

MODEL_NAME="Qwen/Qwen3.5-9B-Base"
DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE_2024="$DATA_DIR/aime-2024-cleaned.parquet"
TEST_FILE_2025="$DATA_DIR/aime-2025-cleaned.parquet"
LOGGER="wandb"  # change to "console" to print to stdout

# Colocated by default: training and inference share the same GPUs. For a
# disaggregated (non-colocated) run, set COLOCATE_ALL=false and split the GPUs,
# e.g. trainer.placement.policy_num_gpus_per_node=4 with the remaining GPUs
# given to the inference engines via generator.inference_engine.num_engines.
COLOCATE_ALL=${COLOCATE_ALL:-true}

NUM_NODES=1
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1

MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=1

# Qwen3.5 goes through the VL bridge (Qwen3VLModel), which packs sequences in its own
# forward and conflicts with SkyRL sample packing; language_model_only routes it to the
# native GPTModel + GDN THD packing path on both the trainer and vLLM.
LANGUAGE_MODEL_ONLY=true

# The default consistent_hash routing sends all samples of a prompt to one engine, so prompts
# that run to the 8192-token cap pile up there and that engine sets the step time. Round robin
# spreads each prompt's samples over all engines.
ROUTER_POLICY=round_robin

# ---- FP8: trainer GEMMs + rollout weight sync ----
# Hopper blockwise FP8 uses exact FP32 block scales end to end. Both exported
# variables are defaulted and validated by SkyRL at startup; set for clarity.
MEGATRON_FP8=e4m3
MEGATRON_FP8_RECIPE=blockwise
MEGATRON_FP8_AMAX_COMPUTE_ALGO=most_recent
MEGATRON_TP_ONLY_AMAX_RED=false
FP8_WEIGHT_SYNC_MODE=blockwise
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export VLLM_USE_DEEP_GEMM_E8M0=0

# ---- Modules kept in BF16 ----
# Training and rollout are configured separately and SkyRL does not check one against the
# other, so to keep a module in BF16 everywhere, list it on both sides.
#
# Training (Megatron). Layer ranges use the layer-count fields, which also work under pipeline
# parallelism. Module globs match Megatron names such as decoder.layers.3.self_attention.linear_qkv
# (GDN layers use self_attention.in_proj / out_proj instead). For finer control, such as a
# different recipe per module, use megatron_config.te_precision_config_file in place of
# fp8_exclude_modules (the two cannot be combined).
NUM_LAYERS_AT_START_IN_BF16=2
NUM_LAYERS_AT_END_IN_BF16=1
MEGATRON_FP8_EXCLUDE_MODULES="['*.self_attention.linear_qkv','*.self_attention.linear_proj']"
# Rollout (vLLM). Globs match HF checkpoint names such as
# model.language_model.layers.3.self_attn.q_proj (GDN layers use linear_attn.* instead). SkyRL
# builds both vLLM's unquantized-module list and the weight sync from this one list, and raises
# at startup if a pattern matches nothing.
FP8_WEIGHT_SYNC_EXCLUDE_MODULES="['*.layers.0.*','*.layers.1.*','*.layers.31.*','*.self_attn.q_proj','*.self_attn.k_proj','*.self_attn.v_proj','*.self_attn.o_proj']"

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE_2024','$TEST_FILE_2025']" \
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
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.ref.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
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
  trainer.policy.megatron_config.num_layers_at_start_in_bf16=$NUM_LAYERS_AT_START_IN_BF16 \
  trainer.ref.megatron_config.num_layers_at_start_in_bf16=$NUM_LAYERS_AT_START_IN_BF16 \
  trainer.policy.megatron_config.num_layers_at_end_in_bf16=$NUM_LAYERS_AT_END_IN_BF16 \
  trainer.ref.megatron_config.num_layers_at_end_in_bf16=$NUM_LAYERS_AT_END_IN_BF16 \
  trainer.policy.megatron_config.fp8_exclude_modules="$MEGATRON_FP8_EXCLUDE_MODULES" \
  trainer.ref.megatron_config.fp8_exclude_modules="$MEGATRON_FP8_EXCLUDE_MODULES" \
  generator.inference_engine.fp8_weight_sync_mode=$FP8_WEIGHT_SYNC_MODE \
  generator.inference_engine.fp8_weight_sync_exclude_modules="$FP8_WEIGHT_SYNC_EXCLUDE_MODULES" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.inference_engine.router_init_kwargs.policy=$ROUTER_POLICY \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=8 \
  trainer.epochs=20 \
  trainer.max_training_steps=100 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=25 \
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
  trainer.run_name="fp8_hopper_blockwise_mixed_qwen35_9b" \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=null \
  trainer.max_ckpts_to_keep=3 \
  $@
