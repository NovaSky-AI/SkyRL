set -x

# Colocated sync DAPO for GLM-5.3 (753B glm_moe_dsa), full fine-tuning: bf16 Megatron trainer,
# online-FP8 vLLM rollouts. 8 nodes x 8xB200, all colocated. Stops after MAX_TRAINING_STEPS.
#
#   bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   export WANDB_API_KEY=<key>
#   bash examples/train/glm5_3/run_dapo_glm5p3_fullft_fp8_rollout_8node.sh
#
# The DAPO budget and algorithm knobs follow
# examples/train/glm5_3_flash/run_dapo_glm5p3_flash_fullft_sync_8node.sh (reasoning there); the
# GLM-5.3-specific settings follow run_gsm8k_glm5p3_lora_fp8_rollout_8node.sh in this directory:
#   * FP8 rollouts without fp8_weight_sync_mode: the trainer syncs bf16 weights and vLLM's
#     layerwise reload re-quantizes each layer as it arrives. One TP8 engine holds ~89 GiB/GPU
#     of FP8 weights; bf16 (~176 GiB) would not fit a B200.
#   * mlp_chunks_for_training=1 (64 deadlocks glm_moe_dsa under EP, see the GSM8K recipe).
#   * Trainer TP8: the colocated weight sync is the memory peak (vLLM ~89 GiB weights + reload
#     buffers + the trainer's bf16 weights and export buffers on the same GPU).

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/models/glm5p3-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"

NUM_NODES=8
NUM_GPUS_PER_NODE=8
# One TP8 FP8 engine per node by default. BF16 rollouts (VLLM_QUANTIZATION=none, ~94 GiB/GPU of
# weights at TP16) need two nodes per engine: NUM_INFERENCE_ENGINES=4
# INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=16, which runs vLLM on its Ray executor (mp is single-node).
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-8}"
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE="${INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE:-8}"
MULTINODE_ENGINE_KWARGS=''
if [ "$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE" -gt "$NUM_GPUS_PER_NODE" ]; then
  INFERENCE_EXECUTOR_BACKEND="${INFERENCE_EXECUTOR_BACKEND:-ray}"
  # A TP group across nodes without multi-node NVLink (B200 over RoCE), on vLLM 0.30:
  # * custom all-reduce rendezvous for an MNNVL buffer that never forms -> engine init deadlocks;
  # * FlashInfer all-reduce and NCCL symmetric memory cannot set up across nodes;
  # * FlashInfer's TRT-LLM sparse-MLA decode has no kernel for 4 heads/rank (64 heads / TP16),
  #   FlashMLA pads heads;
  # * the per-layer FlashInfer fused all-reduce + RMSNorm retries its MNNVL workspace (30 s
  #   timeout) on every call -- skipped by SkyRL's patch_multinode_fused_allreduce_norm.
  MULTINODE_ENGINE_KWARGS='"disable_custom_all_reduce": true, "attention_config": {"backend": "FLASHMLA_SPARSE"}, '
  export VLLM_ALLREDUCE_USE_FLASHINFER=0
  export VLLM_ALLREDUCE_USE_SYMM_MEM=0
else
  INFERENCE_EXECUTOR_BACKEND="${INFERENCE_EXECUTOR_BACKEND:-mp}"
fi
LOGGER="${LOGGER:-wandb}"

MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-20}"

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
INFERENCE_ENGINE_MAX_MODEL_LEN=10752         # prompt + response + chat-template headroom
OVERLONG_BUFFER_LEN=2048                     # penalty starts at 6144
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# dp = 64/TP8 = 8; (policy_mini_batch_size * n_samples) = 384 divides it.
TRAIN_BATCH_SIZE=128
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=12
EVAL_N_SAMPLES_PER_PROMPT=12
MAX_TOKENS_PER_MICROBATCH=8192  # must hold one full sequence

# R3 (rollout router replay); see MOE_BACKEND_KWARG below for the vLLM backend it requires.
ENABLE_ROUTING_REPLAY="${ENABLE_ROUTING_REPLAY:-false}"

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
LR=1e-6

MEGATRON_TP="${MEGATRON_TP:-8}"
MEGATRON_PP=1
MEGATRON_CP=1
# 256 routed experts over EP=64: 4 per GPU, ~23 GiB of bf16 params plus ~45 GiB of fp32 main
# grads. Adam state is on the CPU (~1.1 TB per node).
MEGATRON_EP="${MEGATRON_EP:-64}"
MEGATRON_ETP=1
MLP_CHUNKS_FOR_TRAINING=1
# Megatron-Bridge's GLM5Bridge turns on the DSA indexer KL loss (dsa_indexer_loss_coeff=0.001),
# whose unfused backward materializes attention-sized score tensors and OOMed the first training
# step at 8k responses. Off, as in the GLM-5.3-Flash bridge: the indexer's top-k is not
# differentiable, so the indexer then receives no gradient and stays at its pretrained weights.
DSA_INDEXER_LOSS_COEFF=0.0
# Fused TileLang DSA kernels (indexer top-k and absorbed sparse MLA, fwd+bwd). megatron-core's
# default ("none") computes the indexer's q.k^T for all 32 heads in fp32 before reducing them:
# ~10 GiB per 9k-token sequence, which OOMed the fourth mini-batch of the first step. On B200 the
# TileLang top-k matches the naive one exactly and sparse attention is within bf16 rounding
# (0.3% on out/dq/dk), at ~0.6-1 GiB instead of ~18-19 GiB and ~10x faster. The kernels JIT with
# $CUDA_HOME (CUDA 13.0 here); the cudnn backend would additionally need flash_mla.
#
# DSA_KERNEL_BACKEND=hybrid (default) runs sparse attention on cuDNN/FlashMLA and the indexer top-k
# on TileLang (patch_dsa_hybrid_indexer): -26% step time vs tilelang in the throughput sweep
# (throughput/README.md). FlashMLA comes with the megatron extra (x86_64).
DSA_KERNEL_BACKEND="${DSA_KERNEL_BACKEND:-hybrid}"
if [ "$DSA_KERNEL_BACKEND" = "hybrid" ]; then
  DSA_KERNEL_BACKEND=cudnn
  export SKYRL_DSA_INDEXER_BACKEND=tilelang
fi

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0
# With CPU offload, the default (non-precision-aware) distributed optimizer still builds fp32
# master copies of every local parameter on the GPU (~50 GiB/GPU here) and hands them to the
# hybrid optimizer, which keeps its own pinned fp32 CPU masters to step on. Precision-aware hands
# it the bf16 shards instead, so the fp32 masters live only on the CPU. Without it the trainer sat
# at ~162 GiB allocated and OOMed in a MoE dispatch at step 4. Its known caveat is checkpoint
# saving (megatron-lm#1820); checkpoints are off here.
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-true}"
# DAPO keeps ~192 sequences per engine in flight, so capture decode graphs that far.
INFERENCE_ENGINE_MAX_NUM_SEQS=256
MAX_CUDAGRAPH_CAPTURE_SIZE=256
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.8}"

ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'
# fp8_per_block needs SkyRL's per_block_cast_to_fp8 patch (installed in every vLLM worker); fp8
# is per-tensor. load_format=dummy: SkyRL syncs the trainer's weights before the first rollout.
# VLLM_QUANTIZATION=none serves BF16 (see NUM_INFERENCE_ENGINES above).
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8_per_block}"

# FULL_FP8=true: MXFP8 GEMMs on the trainer (fp8_recipe=auto resolves to MXFP8 on Blackwell) and
# fp8_weight_sync_mode=blockwise, as in examples/train/fp8/run_fp8_blackwell_mxfp8_qwen35_*.sh.
# The trainer casts the bf16 export to 128x128 blockwise FP8 (power-of-2 scales, consumed by
# DeepGEMM as E8M0) using the GLM5 ModelFp8Spec, and SkyRL configures vLLM for checkpoint-format
# FP8 itself (quantization=fp8 + quantization_config), so the online VLLM_QUANTIZATION is unused.
# Primary weights stay bf16: fp8_param is not supported on the MXFP8 path yet.
FULL_FP8="${FULL_FP8:-false}"
FP8_OVERRIDES=()
if [ "$FULL_FP8" = "true" ]; then
  QUANT_KWARG=''
  # blockwise (128x128 FP32-scale blocks, vLLM fp8) or mxfp8 (E8M0 per 32-group, the trainer's
  # own recipe; vLLM serves it through compressed-tensors).
  FP8_WEIGHT_SYNC_MODE="${FP8_WEIGHT_SYNC_MODE:-blockwise}"
  QUANT_LABEL=mxfp8_${FP8_WEIGHT_SYNC_MODE}
  export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=0
  export VLLM_USE_DEEP_GEMM_E8M0=1
  FP8_OVERRIDES=(
    trainer.policy.megatron_config.fp8=e4m3
    trainer.policy.megatron_config.fp8_recipe=auto
    trainer.policy.megatron_config.fp8_amax_compute_algo=most_recent
    trainer.policy.megatron_config.transformer_config_kwargs.tp_only_amax_red=false
    generator.inference_engine.fp8_weight_sync_mode=$FP8_WEIGHT_SYNC_MODE
  )
else
  if [ "$VLLM_QUANTIZATION" = "none" ]; then
    QUANT_KWARG=''
    QUANT_LABEL=bf16
  else
    QUANT_KWARG='"quantization": "'"$VLLM_QUANTIZATION"'", '
    QUANT_LABEL=$VLLM_QUANTIZATION
  fi
fi
# R3 on vLLM's default FP8 MoE backend on B200 (FlashInfer TRT-LLM, monolithic) needs SkyRL's
# patch_routed_experts_rebind (backport of vllm#59455, installed in every vLLM worker): without it
# the capture callback is lost when a weight sync rebuilds the kernel, the capture returns the
# startup profile run's routing for every prefill token, and R3 made the rollout/train logprob gap
# WORSE (0.063 -> 0.164). R3_MOE_BACKEND=deep_gemm (router-side capture, unaffected by the bug)
# is the fallback; it costs ~1.7x generation time.
R3_MOE_BACKEND="${R3_MOE_BACKEND:-auto}"
if [ "$ENABLE_ROUTING_REPLAY" = "true" ] && [ "$R3_MOE_BACKEND" != "auto" ]; then
  MOE_BACKEND_KWARG='"moe_backend": "'"$R3_MOE_BACKEND"'", '
else
  MOE_BACKEND_KWARG=''
fi
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', '"$MOE_BACKEND_KWARG$QUANT_KWARG$MULTINODE_ENGINE_KWARGS"'"load_format": "dummy", "kv_cache_dtype": "bfloat16", "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "max_cudagraph_capture_size": '"$MAX_CUDAGRAPH_CAPTURE_SIZE"', "pass_config": {"fuse_allreduce_rms": false}}}'

export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=128
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
# Ray gives each policy actor num_cpus=1, so OMP_NUM_THREADS would default to 1 and the
# CPU-offloaded Adam step would run single-threaded; 12 halves it (B200 nodes: 14 cores/GPU).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export SKYRL_VLLM_START_PORT="${SKYRL_VLLM_START_PORT:-8400}"
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1

RUN_NAME="${RUN_NAME:-glm5p3_dapo_fullft_${QUANT_LABEL}_tp${MEGATRON_TP}_ep${MEGATRON_EP}_r3${ENABLE_ROUTING_REPLAY}}"
# A full checkpoint is several TB (bf16 weights plus fp32 master weights and Adam state), so
# saving is off by default.
CKPT_INTERVAL="${CKPT_INTERVAL:-0}"
CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/$RUN_NAME}"

${UV_RUN:-uv run --isolated} --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.mtp_num_layers=0 \
  trainer.policy.megatron_config.moe_grouped_gemm=true \
  trainer.policy.megatron_config.moe_token_dispatcher_type="alltoall" \
  trainer.policy.megatron_config.moe_router_score_function="sigmoid" \
  trainer.policy.megatron_config.moe_router_load_balancing_type="none" \
  trainer.policy.megatron_config.moe_enable_routing_replay=$ENABLE_ROUTING_REPLAY \
  generator.inference_engine.enable_return_routed_experts=$ENABLE_ROUTING_REPLAY \
  trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel=true \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="selective" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[core_attn,moe] \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null \
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=$MLP_CHUNKS_FOR_TRAINING \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_indexer_loss_coeff=$DSA_INDEXER_LOSS_COEFF \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_kernel_backend=$DSA_KERNEL_BACKEND \
  trainer.policy.megatron_config.transformer_config_kwargs.disable_parameter_transpose_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$USE_PRECISION_AWARE_OPTIMIZER \
  trainer.policy.optimizer_config.lr=$LR \
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
  trainer.max_training_steps=$MAX_TRAINING_STEPS \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.ckpt_interval=$CKPT_INTERVAL \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.distributed_executor_backend="$INFERENCE_EXECUTOR_BACKEND" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.max_num_seqs=$INFERENCE_ENGINE_MAX_NUM_SEQS \
  generator.inference_engine.gpu_memory_utilization=$INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.router_init_kwargs="$ROUTER_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm5p3_dapo" \
  trainer.run_name="$RUN_NAME" \
  "${FP8_OVERRIDES[@]}" \
  "$@"
