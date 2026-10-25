set -x

# Colocated sync DAPO training+generation for DeepSeek-V4-Flash with Megatron + LoRA, R3 on.
# 4 nodes x 8xB200 (178 GiB/GPU), all colocated, against a running Ray cluster (`ray start` on every node
# with the uv env's Ray version; MODEL_PATH and DATA_DIR present on every node).
#
#   bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   export WANDB_API_KEY=<key>
#   bash examples/train/deepseek_v4_flash/run_dapo_dsv4_flash_lora_sync_4node.sh
#
# Stock DAPO 2k prompt / 8k response budget and algorithm knobs (as in
# examples/train/glm5_3_flash/run_dapo_glm5p3_flash_lora_sync_2node.sh), with DeepSeek-V4-Flash's settings from
# run_gsm8k_dsv4_flash_lora_1node.sh. 4 nodes rather than 2 because training is kernel-launch-bound at this
# pin (mHC Sinkhorn iterations, compressor and indexer per layer on unpacked microbatches), so data
# parallelism is what buys throughput; EP=32 also halves the frozen expert shard (~68 GiB/GPU base).
#
# MODEL_PATH is the full BF16 copy (see run_gsm8k_dsv4_flash_lora_1node.sh for how to build it).

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/avigyabb/models/DeepSeek-V4-Flash-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"

NUM_NODES=4
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=4          # one engine per node, colocated with that node's policy shard
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="${LOGGER:-wandb}"

MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-20}"

MAX_PROMPT_LENGTH=2048
# 8K responses OOM in the step-1 policy forward (146.7 GiB allocated by the trainer + 13.5 GiB held by the
# sleeping vLLM engine on a 178 GiB B200), so the default is 4K; AIME-2024 baseline responses average ~1.6K.
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
INFERENCE_ENGINE_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 512))  # + chat-template headroom
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-$((MAX_RESPONSE_LENGTH / 4))}"  # penalty starts this far before the cap
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# validate_cfg requires (policy_mini_batch_size * n_samples_per_prompt) % dp == 0; dp = 32 (TP=1).
TRAIN_BATCH_SIZE=128
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=12
EVAL_N_SAMPLES_PER_PROMPT=12

# Rollout router replay (R3): vLLM returns the experts it routed each token to and Megatron replays
# that routing. The doc's Stage 2 recipe turns it on for both LoRA and full FT.
ENABLE_ROUTING_REPLAY="${ENABLE_ROUTING_REPLAY:-true}"

# DAPO algorithm knobs
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
LR=1e-5

LORA_RANK=64
LORA_ALPHA=64
MERGE_LORA=false
# Same module lists the Stage 0 LoRA roundtrip row validated.
LORA_TARGET_MODULES='[linear_q_down_proj,linear_q_up_proj,linear_kv_proj,linear_proj,linear_fc1,linear_fc2]'
VLLM_LORA_TARGET_MODULES='["fused_wqa_wkv", "wq_b", "wo_b", "gate_up_proj", "down_proj", "experts"]'

# DSv4 attention is TP=1-only at the pinned megatron-core; EP shards the experts (8/GPU).
MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=32
MEGATRON_ETP=1

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.6}"
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'
# Prefix caching stays on: R3 returns routed experts through vLLM's AuxOutput connector, which requires it.
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "lora_target_modules": '"$VLLM_LORA_TARGET_MODULES"'}'

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
unset NCCL_NET NCCL_TUNER_CONFIG_PATH
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1800
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-${TMPDIR:-/tmp}/vllm_autotune_$(date +%s)}"

RUN_NAME="${RUN_NAME:-dsv4_flash_dapo_sync_lora_r${LORA_RANK}_resp${MAX_RESPONSE_LENGTH}_ep${MEGATRON_EP}}"
CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/$RUN_NAME}"

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
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
  trainer.policy.megatron_config.moe_enable_routing_replay=$ENABLE_ROUTING_REPLAY \
  generator.inference_engine.enable_return_routed_experts=$ENABLE_ROUTING_REPLAY \
  `# Selective recompute (DSv4 attention rejects core_attn checkpointing, mHC rejects full recompute). moe alone` \
  `# OOMs on 8k-token rows; mhc/layernorm under LoRA relies on patch_checkpoint_without_output_frozen_inputs.` \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="selective" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[moe,mhc,layernorm] \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null \
  `# No mlp_chunks_for_training: megatron-core then routes each sequence chunk separately, but R3 hands every` \
  `# chunk the whole microbatch's replayed routes (hash-routed layers fail "expected index [1280, 6] ... [20, 256]").` \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.disable_parameter_transpose_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  `# the full CPU optimizer offload turns this off by default; without it NCCL OOMs lazily allocating buffers for the post-backward metrics all-reduce.` \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.model.lora.target_modules="$LORA_TARGET_MODULES" \
  `# Per-user adapter dir (the default /tmp/skyrl_lora_sync can belong to another user on a shared node). With` \
  `# experts adapted the adapter is ~20 GB per sync, so point LORA_SYNC_PATH at big local storage on every node.` \
  trainer.policy.model.lora.lora_sync_path="${LORA_SYNC_PATH:-${TMPDIR:-/tmp}/skyrl_lora_sync_${USER}}" \
  trainer.policy.megatron_config.lora_config.merge_lora=$MERGE_LORA \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  `# No packed (THD) sequences for DSv4 attention at the pinned megatron-core.` \
  trainer.remove_microbatch_padding=false \
  trainer.use_expandable_segments=true \
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
  trainer.eval_interval=10 \
  trainer.ckpt_interval=10 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
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
  trainer.project_name="dsv4_flash_dapo" \
  trainer.run_name="$RUN_NAME" \
  "$@"
