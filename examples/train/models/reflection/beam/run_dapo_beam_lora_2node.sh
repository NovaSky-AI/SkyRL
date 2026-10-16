set -x

# Colocated synchronous DAPO with LoRA (rank 64) for Beam (501B-A23B, bf16) on AIME-style math.
# Hardware: 2 nodes x 8 B200 (16 GPUs), Megatron trainer + vLLM rollouts sharing the same GPUs. The minimal LoRA
# setup: same layout per GPU as run_dapo_beam_lora_4node.sh with half the batch size (same mini-batch size).
#
#   MODEL_PATH=/path/to/beam ENV_FILE=/path/to/.env bash examples/train/models/reflection/beam/run_dapo_beam_lora_2node.sh
#
# See examples/train/models/reflection/beam/README.md for the environment, data preparation and multi-node setup.
#
# LoRA / trainer optimizations enabled in this recipe:
#   1. vLLM MoE expert LoRA slots at the expert rank (additional_config.max_expert_lora_rank; SkyRL patch).
#   2. Adapter copy + receive buffer moved to pinned CPU after activation: SKYRL_LORA_STAGE_DEVICE=gpu_offload.
#   3. LoRA SP input regather (no gathered LN output kept for backward): lora_config.sequence_parallel_input_regather.
#   4. Fused LM head logprobs with logit softcapping: trainer.fused_lm_head_logprob.

# Hugging Face repo id, or a local copy of the checkpoint at the same path on every node.
MODEL_PATH="${MODEL_PATH:-reflection/Beam-1.0-501B-A23B}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k-cleaned.parquet}"
TEST_FILE="${TEST_FILE:-['$DATA_DIR/aime-2024-cleaned.parquet','$DATA_DIR/aime-2025-cleaned.parquet']}"
LOGGER="${LOGGER:-wandb}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-50}"

# ---- Cluster layout ----
NUM_NODES=2
NUM_GPUS_PER_NODE=8
# One vLLM engine, TP8 x PP2 (one node per PP stage).
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
INFERENCE_ENGINE_PIPELINE_PARALLEL_SIZE=2
INFERENCE_ENGINE_MAX_NUM_SEQS=256
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.7}"
# Megatron: TP4 / PP2 / EP8 -> DP2 on 16 GPUs.
MEGATRON_TP=4
MEGATRON_PP=2
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

# ---- Sequence budget: stock DAPO 2k prompt / 8k response ----
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
INFERENCE_ENGINE_MAX_MODEL_LEN=10752         # prompt + response + chat-template headroom
MAX_TOKENS_PER_MICROBATCH=10752              # one full sequence
OVERLONG_BUFFER_LEN=2048                     # penalty starts at 6144
OVERLONG_BUFFER_PENALTY_FACTOR=1.0
# Beam's chat template defaults reasoning_effort to "max" when unset. Levels: minimal/low/medium/high/xhigh/max.
# Passed through chat_template_kwargs, which requires non-batched generation (generator.batched=false).
REASONING_EFFORT="${REASONING_EFFORT:-minimal}"

# ---- Batch shape ----
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32                # same mini-batch as the 4-node recipe: one optimizer step per training step
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=8

# ---- DAPO algorithm ----
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=1.0
LR=1e-5
# Rollout router replay (R3).
ENABLE_ROUTING_REPLAY=true

# ---- LoRA ----
LORA_RANK=64
LORA_ALPHA=64
LORA_TARGET_MODULES="all-linear"
MERGE_LORA=false
SHARE_EXPERT_ADAPTERS=false        # one adapter per expert
NORMALIZE_MOE_LORA=true            # expert adapters use rank // top_k
MOE_ROUTER_TOPK=8                  # Beam top-k
EXPERT_LORA_RANK=$((LORA_RANK / MOE_ROUTER_TOPK))
# (3) Keep only the sequence-local LN output for LoRA-A backward; re-gather it there.
SEQUENCE_PARALLEL_INPUT_REGATHER=true
# (4) Fused LM head logprobs (softcap-aware).
FUSED_LM_HEAD_LOGPROB=true
OPTIMIZER_OFFLOAD=false            # LoRA optimizer state is small

# ---- vLLM ----
# vLLM-side LoRA targets (merge_lora=false). List both attention layouts: vLLM silently skips adapter
# weights for modules left unwrapped. "experts" covers the MoE.
VLLM_LORA_TARGET_MODULES='["qkv_gate_proj", "qkv_proj", "output_gate_proj", "o_proj", "gate_up_proj", "down_proj", "experts"]'
# (1) Size MoE expert LoRA slots at EXPERT_LORA_RANK instead of max_lora_rank; SkyRL installs the patch in
# each vLLM worker when additional_config.max_expert_lora_rank is set.
ENGINE_INIT_KWARGS='{"disable_custom_all_reduce": false, "max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "lora_target_modules": '"$VLLM_LORA_TARGET_MODULES"', "additional_config": {"max_expert_lora_rank": '"$EXPERT_LORA_RANK"'}}'
# The client fans out one request per sequence; size the router queue past the batch.
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'

# ---- Environment ----
# (2) Build the adapter on GPU, then move the LoRAModel + received tensors to pinned host memory.
export SKYRL_LORA_STAGE_DEVICE=gpu_offload
# DP ranks finish their micro-batches unevenly; early ranks wait in a collective.
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=256
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# Loading a 501B checkpoint into multi-node engines takes longer than the 600 s default.
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S="${SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S:-1800}"
# Surface vLLM worker errors in the driver log.
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1

RUN_NAME="${RUN_NAME:-beam_dapo_lora_r${LORA_RANK}_8k_effort${REASONING_EFFORT}_2node}"
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
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="selective" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[core_attn,moe] \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.model.lora.target_modules="$LORA_TARGET_MODULES" \
  trainer.policy.model.lora.sync_mode=memory \
  trainer.policy.model.lora.share_expert_adapters=$SHARE_EXPERT_ADAPTERS \
  trainer.policy.megatron_config.lora_config.merge_lora=$MERGE_LORA \
  trainer.policy.megatron_config.lora_config.normalize_moe_lora=$NORMALIZE_MOE_LORA \
  trainer.policy.megatron_config.lora_config.sequence_parallel_input_regather=$SEQUENCE_PARALLEL_INPUT_REGATHER \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.remove_microbatch_padding=true \
  trainer.use_expandable_segments=true \
  trainer.fused_lm_head_logprob=$FUSED_LM_HEAD_LOGPROB \
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
  trainer.ckpt_interval="${CKPT_INTERVAL:-10}" \
  trainer.resume_mode="${RESUME_MODE:-null}" \
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
