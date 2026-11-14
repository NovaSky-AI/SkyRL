set -x

# DAPO full fine-tuning of Inkling2 (~594B) colocated on 4 nodes of 8xB200: Megatron (TP2, EP32) and
# four TP8 vLLM engines share all 32 GPUs, taking turns (vLLM sleeps at level 2 while Megatron
# trains; Megatron is offloaded while vLLM generates).
#
#   DATA_DIR=$HOME/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   MODEL_PATH=/path/to/Inkling2 OPTIMIZER_STATE_DIR=/mnt/nvme/skyrl_optimizer_state \
#     bash examples/train/models/thinkingmachines/inkling2/run_dapo_inkling2_fullft_colocated.sh
#
# Memory. Each GPU holds 1/32 of the model in Megatron (~18B params: 37 GB bf16 params plus a 37 GB
# bf16 grad buffer, grad_reduce_in_fp32=false). The optimizer state (fp32 masters and bf16 Adam
# moments, 138 GiB per GPU) lives off the GPU and is streamed through it during each optimizer step
# (megatron_config.optimizer_offload_to_disk). OPTIMIZER_STORAGE=host (default) keeps as much of it as
# fits in page-locked RAM, leaving OPTIMIZER_HOST_RESERVE of RAM free for the params offloaded during
# generation; the rest stays in files under OPTIMIZER_STATE_DIR on NVMe (~half of it on 2 TiB nodes).
# Each vLLM engine holds the model at ~141 GiB/GPU, so the trainer's params and the woken engine weights
# do not fit together: the sync exports from the offloaded params
# (megatron_config.weight_sync_from_offloaded_params).
#
# ROLLOUT_EXPERTS=mxfp8 (default) serves the experts with the vLLM branch's MXFP8 kernels, quantized from
# the synced BF16 weights; bf16 serves them in BF16 (slower generation, smaller rollout-training gap).
#
# Budget and reward shaping as in run_dapo_inkling2_lora_colocated.sh: reasoning effort 0.1, a 4k
# response budget whose last 2k is DAPO's overlong buffer, truncated responses kept in the loss.
#
# Step times and results: docs/content/docs/models/inkling2.mdx. Memory is tight: after a
# sync each GPU has ~9-16 GiB free for the KV-cache wake.
#
# The release's weight matrices all have std exactly 1/sqrt(fan_in) (a norm-constrained optimizer):
# plain Adam drifts them off that manifold, so keep the LR small and watch the rollout/train
# logprob gap and grad norm.

MODEL_PATH="${MODEL_PATH:-thinkingmachines/Inkling-2-Small}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k-cleaned.parquet}"
TEST_FILE="${TEST_FILE:-['$DATA_DIR/aime-2024-cleaned.parquet','$DATA_DIR/aime-2025-cleaned.parquet']}"
export SKYRL_INKLING2_ROLLOUT_EXPERTS="${ROLLOUT_EXPERTS:-${SKYRL_INKLING2_ROLLOUT_EXPERTS:-mxfp8}}"

OPTIMIZER_STATE_DIR="${OPTIMIZER_STATE_DIR:-$HOME/skyrl_optimizer_state}"
OPTIMIZER_MOMENT_DTYPE="${OPTIMIZER_MOMENT_DTYPE:-bf16}"  # 8 instead of 12 B/param streamed per step
OPTIMIZER_STORAGE="${OPTIMIZER_STORAGE:-host}"
OPTIMIZER_HOST_RESERVE="${OPTIMIZER_HOST_RESERVE:-0.35}"  # fraction of RAM kept free when placing state in it
NUM_NODES=4
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="${LOGGER:-wandb}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-50}"

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
INFERENCE_ENGINE_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 512))
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-$((MAX_RESPONSE_LENGTH / 2))}"
OVERLONG_BUFFER_PENALTY_FACTOR=1.0
APPLY_OVERLONG_FILTERING="${APPLY_OVERLONG_FILTERING:-false}"
REASONING_EFFORT="${REASONING_EFFORT:-0.1}"

# 128 prompts x 8 samples per step, 4 optimizer steps of 32 prompts; dp = 32 / TP2 = 16.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-4}"
MAX_TOKENS_PER_MICROBATCH=$INFERENCE_ENGINE_MAX_MODEL_LEN

# REINFORCE (no ratio, no clipping): with score centering below, the gradient is taken against the
# sampler itself, so there is no old-policy forward pass.
POLICY_LOSS_TYPE="${POLICY_LOSS_TYPE:-reinforce}"
LOSS_REDUCTION="token_mean"
USE_KL_LOSS=false
TEMPERATURE=1.0
# Rollout-training mismatch (MXFP8 rollout experts, different kernels) grows with entropy; two replays
# remove its largest parts, and score centering cancels the drift that is left:
# - ROUTER_REPLAY (R3): vLLM returns each token's experts per layer and Megatron routes through them
#   (megatron_config.moe_enable_routing_replay; the fused router is turned off, since it skips replay).
# - SAMPLE_SUPPORT_REPLAY: sample from top-p TOP_P within top-k TOP_K, record that support, and
#   renormalize the trainer's logprobs over it (trainer.algorithm.enable_sample_support_replay), so
#   both sides score the same truncated distribution. Eval samples from the full distribution.
# - SCORE_CENTERING: subtract each token's expected weighted score under the sampler, over the recorded
#   support (trainer.algorithm.enable_score_centering; needs sample-support replay and a loss with no
#   ratio or a rollout-anchored one, such as reinforce).
ROUTER_REPLAY="${ROUTER_REPLAY:-true}"
SAMPLE_SUPPORT_REPLAY="${SAMPLE_SUPPORT_REPLAY:-true}"
SCORE_CENTERING="${SCORE_CENTERING:-$SAMPLE_SUPPORT_REPLAY}"
if [ "$SAMPLE_SUPPORT_REPLAY" = true ]; then
  TOP_P="${TOP_P:-0.95}"
  TOP_K="${TOP_K:-64}"  # must exceed the top-p nucleus
else
  TOP_P="${TOP_P:-1.0}"
  TOP_K="${TOP_K:--1}"
fi
LR="${LR:-1e-6}"

MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=32
MEGATRON_ETP=1

INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.86}"
INFERENCE_ENGINE_MAX_NUM_SEQS="${INFERENCE_ENGINE_MAX_NUM_SEQS:-64}"
# Prefill chunk cap. Each layer's short-convolution state is its own KV-cache group (a 4-token
# window, ~32 KB per token per GPU at TP8: the full hidden width, replicated), and a prefill chunk
# writes a slot for every token in it; with the default chunk one 10k-token request needs ~16 GiB,
# more than the KV room left next to the BF16 weights.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "max_num_batched_tokens": '"$MAX_NUM_BATCHED_TOKENS"'}'
# Weight sync streams the model through one reusable 1 GB CUDA-IPC buffer: larger buffers (or a second
# one) run the trainer out of memory in the EP gather beside the woken engine (~145 GiB/GPU).
WEIGHT_SYNC_BUFFER_GB="${WEIGHT_SYNC_BUFFER_GB:-1}"
# Each engine's TP rank serves 48 of the 384 experts: with weight_sync_local_shard every trainer rank packs
# only the experts of the vLLM worker on its own GPU (per-worker payloads), so a worker receives ~1/8 of
# the experts instead of the whole model.
WEIGHT_SYNC_LOCAL_SHARD="${WEIGHT_SYNC_LOCAL_SHARD:-true}"
WEIGHT_SYNC_NUM_IPC_BUFFERS="${WEIGHT_SYNC_NUM_IPC_BUFFERS:-1}"
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'

export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
# A new communicator's first all-reduce next to nearly full GPUs failed with NVLS on.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
# Multi-node NCCL must run over InfiniBand. On clusters whose image defaults to an OFI/Libfabric net
# plugin, select IB and the compute HCAs explicitly (SkyRL forwards NCCL_* to the workers), e.g.:
#   export NCCL_NET=IB NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_10,mlx5_11
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Checkpoints (CKPT_INTERVAL > 0) are ~1.5 TB per node (params + the disk-offloaded optimizer state), so
# they go to node-local NVMe; rank 0's global files are replicated to every node after a save, and a
# resume (RESUME_MODE=latest) must run on the same nodes.
# MXFP8 training (opt-in, MEGATRON_FP8=e4m3): TE's native Blackwell recipe on the policy's linear-layer
# GEMMs, BF16 primaries and optimizer. The sync still ships the BF16 primaries (there is no FP8 weight
# sync for Inkling2); ROLLOUT_EXPERTS=mxfp8 quantizes the experts on the vLLM side.
FP8_ARGS=()
if [ -n "${MEGATRON_FP8:-}" ]; then
  # MXFP8 needs every GEMM dim divisible by 32; the per-head sink (heads / TP) and the dynamic short-conv
  # kernel projection (kv_heads * width / TP) are smaller, so they stay BF16.
  FP8_ARGS=(
    trainer.policy.megatron_config.fp8=$MEGATRON_FP8
    trainer.policy.megatron_config.fp8_recipe=${MEGATRON_FP8_RECIPE:-mxfp8}
    trainer.policy.megatron_config.fp8_amax_compute_algo=most_recent
    'trainer.policy.megatron_config.fp8_exclude_modules=["*.linear_sink","*.weight_proj"]'
  )
fi

RUN_NAME="${RUN_NAME:-inkling2_dapo_fullft_colocated_lr${LR}_effort${REASONING_EFFORT}_len${MAX_RESPONSE_LENGTH}}"

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="$TEST_FILE" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.algorithm.policy_loss_type=$POLICY_LOSS_TYPE \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.moe_token_dispatcher_type="alltoall" \
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=false \
  trainer.policy.megatron_config.optimizer_offload_to_disk.dir="$OPTIMIZER_STATE_DIR" \
  trainer.policy.megatron_config.optimizer_offload_to_disk.moment_storage_dtype=$OPTIMIZER_MOMENT_DTYPE \
  trainer.policy.megatron_config.optimizer_offload_to_disk.storage=$OPTIMIZER_STORAGE \
  trainer.policy.megatron_config.optimizer_offload_to_disk.host_reserve_fraction=$OPTIMIZER_HOST_RESERVE \
  trainer.policy.megatron_config.weight_sync_from_offloaded_params=true \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="full" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method="uniform" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.remove_microbatch_padding=true \
  trainer.use_expandable_segments=true \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=1 \
  trainer.max_training_steps=$MAX_TRAINING_STEPS \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train="${EVAL_BEFORE_TRAIN:-true}" \
  trainer.eval_interval="${EVAL_INTERVAL:-5}" \
  trainer.ckpt_interval="${CKPT_INTERVAL:--1}" \
  trainer.resume_mode="${RESUME_MODE:-null}" \
  trainer.max_ckpts_to_keep="${MAX_CKPTS_TO_KEEP:-1}" \
  trainer.ckpt_path="${CKPT_PATH:-$HOME/ckpts/$RUN_NAME}" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.weight_transfer_threshold_cuda_ipc_GB=$WEIGHT_SYNC_BUFFER_GB \
  generator.inference_engine.weight_sync_local_shard=$WEIGHT_SYNC_LOCAL_SHARD \
  generator.inference_engine.weight_sync_num_ipc_buffers=$WEIGHT_SYNC_NUM_IPC_BUFFERS \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.max_num_seqs=$INFERENCE_ENGINE_MAX_NUM_SEQS \
  generator.inference_engine.gpu_memory_utilization=$INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.router_init_kwargs="$ROUTER_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.top_k=$TOP_K \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.top_k=-1 \
  trainer.policy.megatron_config.moe_enable_routing_replay=$ROUTER_REPLAY \
  generator.inference_engine.enable_return_routed_experts=$ROUTER_REPLAY \
  trainer.algorithm.enable_sample_support_replay=$SAMPLE_SUPPORT_REPLAY \
  trainer.algorithm.enable_score_centering=$SCORE_CENTERING \
  generator.inference_engine.enable_return_sample_support_set=$SAMPLE_SUPPORT_REPLAY \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=false \
  generator.chat_template_kwargs.reasoning_effort=$REASONING_EFFORT \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="inkling2_dapo" \
  trainer.run_name="$RUN_NAME" \
  "${FP8_ARGS[@]}" \
  "$@"
