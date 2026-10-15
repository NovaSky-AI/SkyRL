set -x

# Trainer-throughput benchmark for GLM-5.3 (753B glm_moe_dsa) full fine-tuning on 8 nodes x
# 8xB200, adapted from examples/train_scripts/full_context/run_full_ctx_megatron.sh. Runs
# NUM_DUMMY_STEPS DAPO-shaped training steps (logprob forward + 4 mini-batch updates) on dummy
# rollouts instead of generating: 128 prompts x 12 samples, prompts 128-512 tokens, responses
# uniform in [512, MAX_RESPONSE_LENGTH] with 25% at the max (truncated), token ids drawn from the
# tokenized DAPO prompts so MoE routing sees real text. Lengths depend only on the step, so every
# configuration trains on the same batches. Results and the sweep are in README.md here.
#
#   bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   bash examples/train/glm5_3/throughput/run_full_ctx_glm5p3_megatron.sh
#
# The defaults are the settings of ../run_dapo_glm5p3_fullft_fp8_rollout_8node.sh. Every knob
# below is an env var; `"$@"` appends arbitrary overrides. SKIP_INFERENCE_ENGINES=true (default)
# benchmarks the trainer alone: no vLLM engines, no weight sync, policy never offloaded. A
# colocated run also keeps a sleeping vLLM engine (~9 GiB/GPU) on every GPU, so keep the
# trainer's peak reserved memory below ~165 GiB for a config to be usable in the real recipe.

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/models/glm5p3-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"

NUM_NODES=8
NUM_GPUS_PER_NODE=8
NUM_DUMMY_STEPS="${NUM_DUMMY_STEPS:-3}"
SKIP_INFERENCE_ENGINES="${SKIP_INFERENCE_ENGINES:-true}"
LOGGER="${LOGGER:-wandb}"

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
TRAIN_BATCH_SIZE=128
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=12
# Soft cap: a longer sequence gets a microbatch of its own.
MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-8192}"

MEGATRON_TP="${MEGATRON_TP:-8}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_EP="${MEGATRON_EP:-64}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-selective}"
RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-[core_attn,moe]}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-null}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-null}"
DSA_KERNEL_BACKEND="${DSA_KERNEL_BACKEND:-tilelang}"
# alltoall, or flex with MOE_FLEX_BACKEND=deepep|hybridep|ncclep.
MOE_DISPATCHER="${MOE_DISPATCHER:-alltoall}"
MOE_FLEX_BACKEND="${MOE_FLEX_BACKEND:-deepep}"
GRAD_REDUCE_IN_FP32="${GRAD_REDUCE_IN_FP32:-true}"
ENABLE_ROUTING_REPLAY=false

# FULL_FP8=true: MXFP8 GEMMs on the trainer, as FULL_FP8 in the DAPO recipe.
FULL_FP8="${FULL_FP8:-false}"
FP8_OVERRIDES=()
if [ "$FULL_FP8" = "true" ]; then
  PRECISION=mxfp8
  export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=0
  FP8_OVERRIDES=(
    trainer.policy.megatron_config.fp8=e4m3
    trainer.policy.megatron_config.fp8_recipe=auto
    trainer.policy.megatron_config.fp8_amax_compute_algo=most_recent
    trainer.policy.megatron_config.transformer_config_kwargs.tp_only_amax_red=false
  )
else
  PRECISION=bf16
fi
DISPATCH_OVERRIDES=( trainer.policy.megatron_config.moe_token_dispatcher_type="$MOE_DISPATCHER" )
if [ "$MOE_DISPATCHER" = "flex" ]; then
  DISPATCH_OVERRIDES+=( trainer.policy.megatron_config.transformer_config_kwargs.moe_flex_dispatcher_backend="$MOE_FLEX_BACKEND" )
  DISPATCH_LABEL="flex_$MOE_FLEX_BACKEND"
else
  DISPATCH_LABEL=$MOE_DISPATCHER
fi
if [ "$SKIP_INFERENCE_ENGINES" = "true" ]; then COLOCATE_ALL=false; else COLOCATE_ALL=true; fi

export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1

RUN_NAME="${RUN_NAME:-glm5p3_fullctx${MAX_RESPONSE_LENGTH}_${PRECISION}_tp${MEGATRON_TP}_ep${MEGATRON_EP}_rc${RECOMPUTE_MODULES//[\[\],]/}_${DSA_KERNEL_BACKEND}_${DISPATCH_LABEL}_mtpm${MAX_TOKENS_PER_MICROBATCH}}"

${UV_RUN:-uv run --isolated} --extra megatron -m examples.train_scripts.full_context.main_full_ctx \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.num_dummy_steps=$NUM_DUMMY_STEPS \
  trainer.skip_inference_engines=$SKIP_INFERENCE_ENGINES \
  trainer.dummy_length_distribution=mixture \
  trainer.dummy_min_prompt_length=128 \
  trainer.dummy_max_prompt_length=512 \
  trainer.dummy_min_response_length=512 \
  trainer.dummy_truncated_fraction=0.25 \
  trainer.dummy_token_source=dataset \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.clip_ratio_c=10.0 \
  trainer.algorithm.loss_reduction="token_mean" \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=$COLOCATE_ALL \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.mtp_num_layers=0 \
  trainer.policy.megatron_config.moe_grouped_gemm=true \
  trainer.policy.megatron_config.moe_router_score_function="sigmoid" \
  trainer.policy.megatron_config.moe_router_load_balancing_type="none" \
  trainer.policy.megatron_config.moe_enable_routing_replay=$ENABLE_ROUTING_REPLAY \
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=$GRAD_REDUCE_IN_FP32 \
  trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel=true \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=$RECOMPUTE_GRANULARITY \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=$RECOMPUTE_MODULES \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=$RECOMPUTE_METHOD \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=$RECOMPUTE_NUM_LAYERS \
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=1 \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_indexer_loss_coeff=0.0 \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_kernel_backend=$DSA_KERNEL_BACKEND \
  trainer.policy.megatron_config.transformer_config_kwargs.disable_parameter_transpose_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.remove_microbatch_padding=true \
  trainer.use_expandable_segments=true \
  trainer.fused_lm_head_logprob=true \
  trainer.logprobs_chunk_size=1024 \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.ckpt_interval=0 \
  trainer.resume_mode=null \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=8 \
  generator.inference_engine.tensor_parallel_size=8 \
  generator.inference_engine.enable_return_routed_experts=$ENABLE_ROUTING_REPLAY \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm5p3_throughput" \
  trainer.run_name="$RUN_NAME" \
  "${DISPATCH_OVERRIDES[@]}" \
  "${FP8_OVERRIDES[@]}" \
  "$@"
