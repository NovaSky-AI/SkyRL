set -x

# Colocated synchronous DAPO, full-parameter fine-tuning of Beam (501B-A23B, bf16) with rollout router replay (R3).
# Hardware: 8 nodes x 8 B200 (64 GPUs) with ~1 TB of free node-local NVMe per node for the optimizer state.
#
#   MODEL_PATH=/path/to/beam ENV_FILE=/path/to/.env bash examples/train/models/reflection/beam/run_dapo_beam_fullft_8node.sh
#
# See examples/train/models/reflection/beam/README.md for the environment, data preparation and multi-node setup.

# Hugging Face repo id, or a local copy of the checkpoint at the same path on every node.
MODEL_PATH="${MODEL_PATH:-reflection/Beam-1.0-501B-A23B}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k-cleaned.parquet}"
TEST_FILE="${TEST_FILE:-['$DATA_DIR/aime-2024-cleaned.parquet','$DATA_DIR/aime-2025-cleaned.parquet']}"

NUM_NODES=8
NUM_GPUS_PER_NODE=8
# vLLM: 4 engines of PP2 x TP8, each spanning a node pair (~63 GB/GPU of bf16 weights). TP stays inside a node.
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
INFERENCE_ENGINE_PIPELINE_PARALLEL_SIZE=2
# Lower than the LoRA recipe's 0.7: full FT leaves more Megatron state on the GPU around weight sync.
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.6}"
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
# dp = 64 / (TP4 * PP4 * CP1) = 4 here: 32 * 8 = 256 is divisible (and so is a TRAIN_BATCH_SIZE=32 smoke test).
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

# Megatron: TP4 (sequence parallel) x PP4 x EP8, ETP1. PP4 gives 13 of the 52 layers per stage; EP8 keeps the MoE
# all-to-all inside a node (NVLink) and puts 32 of the 256 experts on each GPU. Per GPU: ~490B expert params /
# (EP8 * PP4) = ~15B (bf16 ~31 GB + fp32 grad buffer ~61 GB) plus ~1B dense params / (TP4 * PP4) (~5 GB), so ~97 GB
# stays on the GPU during training. Expert data parallel = 64 / (EP8 * ETP1 * PP4) = 2.
MEGATRON_TP="${MEGATRON_TP:-4}"
MEGATRON_PP="${MEGATRON_PP:-4}"
MEGATRON_CP=1
MEGATRON_EP="${MEGATRON_EP:-8}"
MEGATRON_ETP=1

# Optimizer state (fp32 master weights + Adam m/v, 12 B/param, ~6 TB in all). Default: on node-local NVMe
# (megatron_config.optimizer_offload_to_disk): the state lives in files under $OPTIMIZER_STATE_DISK_DIR/$RUN_NAME and
# the optimizer step streams it through the GPU one chunk (200M elements) at a time. Host memory stays at the pinned
# bf16 weight copies (~0.25 TB/node); disk ~0.72 TB/node, read and written every optimizer step, so use fast local
# NVMe (RAID0). Same numerics as the GPU optimizer with fp32 moments; the run's state dir is cleared at startup and
# removed at the end of training.
# OPTIMIZER_STATE_DISK_DIR= (empty) falls back to the CPU optimizer (HybridDeviceOptimizer), which needs ~1.7 TB of
# host memory per node at this layout.
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

# The client fans out one request per sequence, so the router sees the whole batch at once; its
# defaults (queue_size=100, queue_timeout_secs=60) drop the overflow. Size the queue past the batch.
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'
# TP8 stays inside a node, so vLLM's custom all-reduce is usable.
ENGINE_INIT_KWARGS='{"disable_custom_all_reduce": false, "max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"'}'

# NCCL timeout: DP ranks finish their micro-batches unevenly and the early ones sit in a collective.
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=256
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# Loading a 501B checkpoint into multi-node engines takes longer than the 600 s default.
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S="${SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S:-1800}"
# Surface vLLM worker errors in the driver log.
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1

RUN_NAME="${RUN_NAME:-beam_dapo_fullft_8k_effort${REASONING_EFFORT}_tp${MEGATRON_TP}ep${MEGATRON_EP}_r3${ENABLE_ROUTING_REPLAY}_8node}"
# A full checkpoint (bf16 weights + fp32 master weights + Adam state) is ~7 TB in all (~0.9 TB per node).
CKPT_INTERVAL="${CKPT_INTERVAL:-25}"
CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/$RUN_NAME}"
EXPORT_PATH="${EXPORT_PATH:-$HOME/exports/$RUN_NAME}"
LOG_PATH="${LOG_PATH:-$HOME/skyrl-logs}"

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
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.distributed_executor_backend="ray" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.pipeline_parallel_size=$INFERENCE_ENGINE_PIPELINE_PARALLEL_SIZE \
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
