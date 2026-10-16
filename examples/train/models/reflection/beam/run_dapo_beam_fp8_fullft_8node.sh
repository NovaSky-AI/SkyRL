set -x

# Full-FP8 colocated synchronous DAPO, full-parameter fine-tuning of Beam (501B-A23B) with R3: MXFP8 trainer GEMMs,
# MXFP8 rollout weights, bf16 KV cache. Hardware: 8 nodes x 8 B200 (64 GPUs) with ~1 TB of free NVMe per node.
#
#   MODEL_PATH=/path/to/beam ENV_FILE=/path/to/.env bash examples/train/models/reflection/beam/run_dapo_beam_fp8_fullft_8node.sh
#
# See examples/train/models/reflection/beam/README.md for the environment, data preparation and multi-node setup.
#
# Same as run_dapo_beam_fullft_8node.sh (disk-streamed optimizer state, R3, minimal effort) but with Megatron
# TP4/PP8/EP8 (see MEGATRON_PP below), plus:
#   1. trainer GEMMs in TE's MXFP8 recipe (fp8_recipe=mxfp8); primary weights stay bf16 (fp8_param=false,
#      which the disk-streamed optimizer state also requires);
#   2. rollout weights synced as MXFP8 (fp8_weight_sync_mode=mxfp8) from the bf16 checkpoint, so vLLM serves FP8 and
#      trainer and sampler share one representation. The quantize/ignore split comes from BEAM_FP8_SPEC
#      (skyrl/backends/skyrl_train/weight_sync/fp8/models/beam.py);
#   3. a bf16 KV cache (KV_CACHE_DTYPE=auto, the default; fp8_e4m3 for faster saturated decode);
#   4. geometric sequence masking (off_policy_correction, [0.99, 1.01])
# Rollouts: 8 single-node vLLM engines, TP8 + EP8 each (attention/dense TP8, experts EP8), ~61 GiB/GPU of MXFP8 weights.
#
# If a weight sync stalls, check the vLLM worker logs for an OutOfMemoryError in update_weights (an OOM there shows up
# as a hang on the trainer side) and lower INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION.

# Hugging Face repo id, or a local copy of the checkpoint at the same path on every node.
MODEL_PATH="${MODEL_PATH:-reflection/Beam-1.0-501B-A23B}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k-cleaned.parquet}"
TEST_FILE="${TEST_FILE:-['$DATA_DIR/aime-2024-cleaned.parquet','$DATA_DIR/aime-2025-cleaned.parquet']}"

NUM_NODES=8
NUM_GPUS_PER_NODE=8
# vLLM: 8 engines, one per node, TP8 + EP8 (EP = DP x TP = 8): ~61 GiB/GPU of MXFP8 weights. EP is required: plain TP8
# shards each expert's FFN to 320, which the FlashInfer TRT-LLM MXFP8 MoE cannot lay out (it needs a multiple of 128).
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
INFERENCE_ENGINE_PIPELINE_PARALLEL_SIZE=1
INFERENCE_ENGINE_EXPERT_PARALLEL_SIZE=8
# At 0.55 a TP8+EP8 engine keeps ~98 GiB: ~61 GiB weights + ~30 GiB KV (~1.2M tokens, ~110 full-length sequences
# per engine vs ~64 needed per step) + graphs/workspaces, leaving room for Megatron state around weight sync.
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.55}"
INFERENCE_ENGINE_MAX_NUM_SEQS=256
LOGGER="${LOGGER:-wandb}"

MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-50}"

# Sequence budget: the stock DAPO 2k prompt + 8k response. Overlong filtering absorbs the truncated tail.
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
INFERENCE_ENGINE_MAX_MODEL_LEN=10752         # prompt + response + chat-template headroom
OVERLONG_BUFFER_LEN=2048                     # penalty starts at 6144
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# Beam's chat template defaults reasoning_effort to "max" when unset. Levels: minimal/low/medium/high/xhigh/max.
# Passed through generator.chat_template_kwargs, which needs non-batched generation.
REASONING_EFFORT="${REASONING_EFFORT:-minimal}"

# Batch shape. validate_cfg requires (policy_mini_batch_size * n_samples_per_prompt) % dp == 0;
# dp = 64 / (TP4 * PP8 * CP1) = 2 here: 32 * 8 = 256 is divisible (and so is a TRAIN_BATCH_SIZE=32 smoke test).
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-8}"
MAX_TOKENS_PER_MICROBATCH=10752  # must hold one full sequence (prompt + response)

# Rollout router replay (R3): replays the rollout's expert routing in the trainer.
ENABLE_ROUTING_REPLAY="${ENABLE_ROUTING_REPLAY:-true}"

# DAPO algorithm knobs (same as the LoRA recipe)
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=1.0
LR="${LR:-1e-6}"                 # full fine-tuning

# Megatron: TP4 (sequence parallel) x PP8 x EP8, ETP1. EP8 keeps the MoE all-to-all inside a node (NVLink).
MEGATRON_TP="${MEGATRON_TP:-4}"
# PP8, not the bf16 recipe's PP4: TE's FP8 weight copies on top of PP4's ~97 GB of resident bf16 weights + fp32 grads
# do not fit. PP8 peaks at ~118 GB/GPU; the Beam bridge splits the 52 layers 7,7,7,7,6,6,6,6. DP = 64 / (TP4 * PP8) = 2.
MEGATRON_PP="${MEGATRON_PP:-8}"
MEGATRON_CP=1
MEGATRON_EP="${MEGATRON_EP:-8}"
MEGATRON_ETP=1

# Optimizer state (fp32 master weights + Adam m/v, 12 B/param, ~6 TB in all). Default: on node-local NVMe
# (megatron_config.optimizer_offload_to_disk): the state lives in files under $OPTIMIZER_STATE_DISK_DIR/$RUN_NAME and
# the optimizer step streams it through the GPU one chunk (200M elements) at a time. Host memory stays at the pinned
# bf16 weight copies; disk ~0.72 TB/node, read and written every optimizer step, so use fast local NVMe (RAID0).
# Same numerics as the GPU optimizer with fp32 moments; the run's state dir is cleared at startup and removed at the
# end of training.
# OPTIMIZER_STATE_DISK_DIR= (empty) falls back to the CPU optimizer (HybridDeviceOptimizer), which needs ~1.7 TB of
# host memory per node.
OPTIMIZER_STATE_DISK_DIR="${OPTIMIZER_STATE_DISK_DIR-$HOME/skyrl_optimizer_state}"
OPTIMIZER_STATE_DISK_MOMENT_DTYPE="${OPTIMIZER_STATE_DISK_MOMENT_DTYPE:-fp32}"  # bf16 cuts streamed bytes by 1/3
if [ -n "$OPTIMIZER_STATE_DISK_DIR" ]; then
  DISK_OPTIMIZER_ARGS=(trainer.policy.megatron_config.optimizer_offload_to_disk.dir="$OPTIMIZER_STATE_DISK_DIR"
    trainer.policy.megatron_config.optimizer_offload_to_disk.moment_storage_dtype="$OPTIMIZER_STATE_DISK_MOMENT_DTYPE")
  OPTIMIZER_OFFLOAD=false  # the streamed state replaces the CPU optimizer (mutually exclusive)
else
  DISK_OPTIMIZER_ARGS=()
  # The CPU optimizer's host state exceeds one socket: prefer (don't force) GPU-local NUMA memory.
  export SKYRL_NUMA_MEMBIND="${SKYRL_NUMA_MEMBIND:-preferred}"
fi
# CPU optimizer (fallback): builds the fp32 master params itself (precision-aware optimizer), with pageable buffers
# (pinned power-of-two buffers overflow a NUMA node).
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-true}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
PRECISION_AWARE_OPTIMIZER="${PRECISION_AWARE_OPTIMIZER:-$OPTIMIZER_OFFLOAD}"
PIN_CPU_OPTIMIZER="${PIN_CPU_OPTIMIZER:-$( [ "$OPTIMIZER_OFFLOAD" = true ] && echo false || echo true )}"

# Full recompute keeps only each layer's input, leaving headroom for 10K-token micro-batches.
RECOMPUTE_ARGS=(trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="full"
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[core_attn]
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method="uniform"
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1)

# ---- FP8 (trainer GEMMs, rollout weights, KV cache) ----
# MXFP8 (E8M0 scale per 32 elements), TE's native recipe on Blackwell. Set explicitly, not "auto": validate_cfg runs
# on the launching driver, which may have no GPU, where "auto" cannot resolve.
MEGATRON_FP8="${MEGATRON_FP8:-e4m3}"
MEGATRON_FP8_RECIPE="${MEGATRON_FP8_RECIPE:-mxfp8}"
MEGATRON_FP8_AMAX_COMPUTE_ALGO=most_recent
MEGATRON_TP_ONLY_AMAX_RED=false
# FP8 rollout weights via BEAM_FP8_SPEC, on the MXFP8 wire (matches the trainer's recipe). FP8_ROLLOUT=0: FP8 training
# with a bf16 rollout (then use the bf16 recipe's PP2 x TP8 engines; bf16 does not fit one node next to the trainer).
FP8_ROLLOUT="${FP8_ROLLOUT:-1}"
FP8_WEIGHT_SYNC_MODE="${FP8_WEIGHT_SYNC_MODE:-mxfp8}"
if [ "$FP8_ROLLOUT" = 1 ]; then FP8_ROLLOUT_ARGS=(generator.inference_engine.fp8_weight_sync_mode=$FP8_WEIGHT_SYNC_MODE); else FP8_ROLLOUT_ARGS=(); fi
# bf16 KV cache by default; KV_CACHE_DTYPE=fp8_e4m3 trades ~0.01 more rollout/train logprob mismatch for faster saturated decode.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
# Power-of-2 (E8M0) block scales on both ends; SkyRL forwards both variables to the Ray workers.
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES="${NVTE_FP8_BLOCK_SCALING_FP32_SCALES:-0}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-1}"
# Cold FlashInfer autotune on the first MXFP8 MoE prepare can take many minutes.
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S="${SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S:-7200}"

# The client fans out one request per sequence, so the router sees the whole batch at once; its
# defaults (queue_size=100, queue_timeout_secs=60) drop the overflow. Size the queue past the batch.
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'
# TP8 stays inside a node, so vLLM's custom all-reduce is usable.
ENGINE_INIT_KWARGS='{"disable_custom_all_reduce": false, "max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "kv_cache_dtype": "'"$KV_CACHE_DTYPE"'"}'

# NCCL timeout: DP ranks finish their micro-batches unevenly and the early ones sit in a collective.
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=256  # >= the ~64 sequences per engine per step (512 / 8 engines)
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# Surface vLLM worker errors in the driver log.
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1

RUN_NAME="${RUN_NAME:-beam_dapo_fp8_fullft_8k_effort${REASONING_EFFORT}_rollout$( [ "$FP8_ROLLOUT" = 1 ] && echo mxfp8_tp8ep8 || echo bf16 )_kv${KV_CACHE_DTYPE}_8node}"
# A full checkpoint (bf16 weights + fp32 master weights + Adam state) is ~7 TB in all (~0.9 TB per node).
# CKPT_INTERVAL=0 turns saving off.
CKPT_INTERVAL="${CKPT_INTERVAL:-25}"
CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/$RUN_NAME}"
EXPORT_PATH="${EXPORT_PATH:-$HOME/exports/$RUN_NAME}"
LOG_PATH="${LOG_PATH:-$HOME/skyrl-logs}"

# Off-policy correction: The geometric sequence mask drops the few
# sequences whose geometric-mean ratio falls outside [GEO_MASK_LOW, GEO_MASK_HIGH].
# SEQUENCE_MASK_METRIC= (empty) disables the mask (passed as null).
SEQUENCE_MASK_METRIC="${SEQUENCE_MASK_METRIC-geometric}"
[ -z "$SEQUENCE_MASK_METRIC" ] && SEQUENCE_MASK_METRIC=null
GEO_MASK_HIGH=1.01
GEO_MASK_LOW=0.99

# UV_EXTRA_ARGS: optional extra `uv run` flags, e.g. "--with ray==<your cluster Ray version>".
uv run --isolated --extra megatron --extra fa4 ${UV_EXTRA_ARGS} ${ENV_FILE:+--env-file "$ENV_FILE"} \
  -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="$TEST_FILE" \
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
  trainer.policy.megatron_config.moe_grouped_gemm=true \
  trainer.policy.megatron_config.moe_token_dispatcher_type="alltoall" \
  trainer.policy.megatron_config.moe_router_load_balancing_type="none" \
  trainer.policy.megatron_config.moe_enable_routing_replay=$ENABLE_ROUTING_REPLAY \
  generator.inference_engine.enable_return_routed_experts=$ENABLE_ROUTING_REPLAY \
  trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel=true \
  "${RECOMPUTE_ARGS[@]}" \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$PRECISION_AWARE_OPTIMIZER \
  trainer.policy.megatron_config.optimizer_config_kwargs.pin_cpu_params=$PIN_CPU_OPTIMIZER \
  trainer.policy.megatron_config.optimizer_config_kwargs.pin_cpu_grads=$PIN_CPU_OPTIMIZER \
  "${DISK_OPTIMIZER_ARGS[@]}" \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.remove_microbatch_padding=true \
  trainer.use_expandable_segments=true \
  trainer.fused_lm_head_logprob=true \
  trainer.policy.megatron_config.fp8=$MEGATRON_FP8 \
  trainer.policy.megatron_config.fp8_recipe=$MEGATRON_FP8_RECIPE \
  trainer.policy.megatron_config.fp8_amax_compute_algo=$MEGATRON_FP8_AMAX_COMPUTE_ALGO \
  trainer.policy.megatron_config.transformer_config_kwargs.tp_only_amax_red=$MEGATRON_TP_ONLY_AMAX_RED \
  trainer.policy.megatron_config.transformer_config_kwargs.fp8_param=false \
  "${FP8_ROLLOUT_ARGS[@]}" \
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
  trainer.eval_before_train="${EVAL_BEFORE_TRAIN:-true}" \
  trainer.eval_interval="${EVAL_INTERVAL:-5}" \
  trainer.ckpt_interval=$CKPT_INTERVAL \
  trainer.resume_mode="${RESUME_MODE:-latest}" \
  trainer.log_path="$LOG_PATH" \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.algorithm.off_policy_correction.sequence_mask_metric=$SEQUENCE_MASK_METRIC \
  trainer.algorithm.off_policy_correction.geo_mask_high=$GEO_MASK_HIGH \
  trainer.algorithm.off_policy_correction.geo_mask_low=$GEO_MASK_LOW \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.distributed_executor_backend="ray" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.pipeline_parallel_size=$INFERENCE_ENGINE_PIPELINE_PARALLEL_SIZE \
  generator.inference_engine.expert_parallel_size=$INFERENCE_ENGINE_EXPERT_PARALLEL_SIZE \
  generator.inference_engine.max_num_seqs=$INFERENCE_ENGINE_MAX_NUM_SEQS \
  generator.inference_engine.gpu_memory_utilization=$INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.router_init_kwargs="$ROUTER_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=false \
  generator.chat_template_kwargs.reasoning_effort=$REASONING_EFFORT \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="${PROJECT_NAME:-skyrl-beam}" \
  trainer.run_name="$RUN_NAME" \
  "$@"
