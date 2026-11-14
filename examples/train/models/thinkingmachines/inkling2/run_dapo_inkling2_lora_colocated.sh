set -x

# DAPO on DAPO-Math-17k (eval: AIME 2024 + 2025) for Inkling2 with Megatron + LoRA, colocated on
# 4 nodes of 8xB200: Megatron (TP2, EP32) and four TP8 vLLM engines share all 32 GPUs.
#
#   DATA_DIR=$HOME/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   bash examples/train/models/thinkingmachines/inkling2/run_dapo_inkling2_lora_colocated.sh
#
# MODEL_PATH defaults to the release, thinkingmachines/Inkling-2-Small (NVFP4 experts); its MXFP8 variant
# (thinkingmachines/Inkling-2-Small-MXFP8) also works. Megatron-Bridge and vLLM both dequantize the
# experts to BF16 as they load, so trainer and rollouts start from the same BF16 model.
# Adapter-only sync (merge_lora=false) needs the LoRA support of
# the Inkling2 vLLM build (vllm/models/inkling2/lora.py).
#
# Placement. Each engine holds the BF16 model at ~147 GiB/GPU, so the two sides take turns:
# - Generation: Megatron is offloaded. Its frozen base (~37 GB/GPU at EP32) goes to file-backed
#   copies on node NVMe (SKYRL_FROZEN_OFFLOAD_DIR); the LoRA buffers stay resident.
# - Training: the engines sleep at level 1 (LoRA keeps the base weights, so they are backed up to
#   host RAM: ~1.2 TB per node, with ~2 TB per node here).
# - Sync: adapter only, with Megatron offloaded.
#
# Budget. At reasoning effort 0.1 the model already solves ~87% of DAPO-Math with an 8k budget
# (median response 1.3k tokens, 14% past 4k, 6.5% truncated at 8k), so the reward is shaped by
# length: a 4k budget whose last 2k is DAPO's overlong buffer (a penalty growing to 1 at 4k).
# Truncated responses stay in the loss (APPLY_OVERLONG_FILTERING=false), as in the colocated full
# fine-tuning run this recipe is compared against (run_dapo_inkling2_fullft_colocated.sh).

MODEL_PATH="${MODEL_PATH:-thinkingmachines/Inkling-2-Small}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k-cleaned.parquet}"
TEST_FILE="${TEST_FILE:-['$DATA_DIR/aime-2024-cleaned.parquet','$DATA_DIR/aime-2025-cleaned.parquet']}"
# Megatron's offloaded frozen base weights (file-backed, clean page cache): node-local NVMe.
export SKYRL_FROZEN_OFFLOAD_DIR="${SKYRL_FROZEN_OFFLOAD_DIR:-$HOME/skyrl_frozen_offload}"

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

# Inkling2's thinking effort (0-1; the chat template defaults to 0.9). chat_template_kwargs needs
# non-batched generation.
REASONING_EFFORT="${REASONING_EFFORT:-0.1}"

# 128 prompts x 8 samples per step, 4 optimizer steps of 32 prompts. validate_cfg needs
# (mini_batch * n_samples) % dp == 0, and dp = 32 / TP2 = 16.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-4}"
MAX_TOKENS_PER_MICROBATCH=$INFERENCE_ENGINE_MAX_MODEL_LEN  # holds one full sequence

# DAPO
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
LOSS_REDUCTION="token_mean"
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# LoRA on every linear ("all-linear" adds Inkling2's attention projections), synced as an adapter
# and served by vLLM's LoRA. Experts at rank // top_k (normalize_moe_lora: 16 at rank 128), with
# vLLM's expert slots sized to match (scale_moe_lora_rank). SHARE_EXPERT_ADAPTERS=true trains one
# adapter per EP rank, shared by its 12 local experts; false (the default) one per expert.
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SHARE_EXPERT_ADAPTERS="${SHARE_EXPERT_ADAPTERS:-false}"
MERGE_LORA=false
NORMALIZE_MOE_LORA=true
# Adapter sync: "disk" (PEFT files on a path every node sees, loaded by vLLM's directory path) or
# "memory" (CUDA IPC from the colocated trainer).
LORA_SYNC_MODE="${LORA_SYNC_MODE:-memory}"
LORA_SYNC_PATH="${LORA_SYNC_PATH:-$HOME/skyrl_lora_sync}"

MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=32
MEGATRON_ETP=1

# The engines get the GPUs while Megatron is offloaded; leave room for the trainer processes'
# contexts, NCCL buffers and resident LoRA buffers.
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.86}"
# The KV cache holds ~20k tokens next to the BF16 weights (~10-20 running sequences).
INFERENCE_ENGINE_MAX_NUM_SEQS="${INFERENCE_ENGINE_MAX_NUM_SEQS:-64}"
# Prefill chunk cap. Each layer's short-convolution state is its own KV-cache group (a 4-token
# window, ~32 KB per token per GPU at TP8: the full hidden width, replicated), and a prefill chunk
# writes a slot for every token in it; with the default chunk one 10k-token request needs ~16 GiB,
# more than the KV room left next to the BF16 weights.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "max_num_batched_tokens": '"$MAX_NUM_BATCHED_TOKENS"'}'
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'
# vLLM executor ("ray" or "mp"; each TP8 engine is one node).
INFERENCE_ENGINE_EXECUTOR="${INFERENCE_ENGINE_EXECUTOR:-ray}"

export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
# Multi-node NCCL must run over InfiniBand. On clusters whose image defaults to an OFI/Libfabric net
# plugin, select IB and the compute HCAs explicitly (SkyRL forwards NCCL_* to the workers), e.g.:
#   export NCCL_NET=IB NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_10,mlx5_11
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

RUN_NAME="${RUN_NAME:-inkling2_dapo_lora_colocated_r${LORA_RANK}_effort${REASONING_EFFORT}_len${MAX_RESPONSE_LENGTH}}"

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="$TEST_FILE" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.grpo_norm_by_std=true \
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
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="full" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method="uniform" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.model.lora.share_expert_adapters=$SHARE_EXPERT_ADAPTERS \
  trainer.policy.megatron_config.lora_config.merge_lora=$MERGE_LORA \
  trainer.policy.model.lora.sync_mode=$LORA_SYNC_MODE \
  trainer.policy.model.lora.lora_sync_path="$LORA_SYNC_PATH" \
  trainer.policy.megatron_config.lora_config.normalize_moe_lora=$NORMALIZE_MOE_LORA \
  generator.inference_engine.scale_moe_lora_rank=true \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=$WEIGHT_DECAY \
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
  trainer.ckpt_path="${CKPT_PATH:-$HOME/ckpts/$RUN_NAME}" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.distributed_executor_backend=$INFERENCE_ENGINE_EXECUTOR \
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
  generator.eval_sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  `# Unbatched: SkyRL renders the prompt, so the chat template sees reasoning_effort.` \
  generator.batched=false \
  generator.chat_template_kwargs.reasoning_effort=$REASONING_EFFORT \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="inkling2_dapo" \
  trainer.run_name="$RUN_NAME" \
  "$@"
