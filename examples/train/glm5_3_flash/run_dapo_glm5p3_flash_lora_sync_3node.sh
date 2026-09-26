set -x

# Colocated sync DAPO training+generation for GLM-5.3-Flash with Megatron + LoRA.
# 2 nodes x 8xB300, all colocated. Stops after MAX_TRAINING_STEPS.
#
#   bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   export WANDB_API_KEY=<key>
#   bash examples/train/glm5_3_flash/run_dapo_glm5p3_flash_lora_sync_3node.sh
#
# Stock DAPO 2k prompt / 8k response budget with merge_lora=false LoRA sync. Per-setting
# reasoning lives in run_gsm8k_glm5p3_flash_lora_1node.sh.

MODEL_PATH="${MODEL_PATH:-/data/trajectory/model-cache/glm5p3-flash-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"

NUM_NODES=2
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=2          # one engine per node, colocated with that node's policy shard
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="${LOGGER:-wandb}"

MAX_TRAINING_STEPS=50

# Sequence budget: the stock DAPO 2k prompt + 8k response. Sequences past
# dsa_indexer_topk=2048 are handled by megatron-core's k-pool indexer (NVIDIA/Megatron-LM#7054).
# Overlong filtering absorbs the truncated tail. ~40 min/step at this size.
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
INFERENCE_ENGINE_MAX_MODEL_LEN=10752         # prompt + response + chat-template headroom
OVERLONG_BUFFER_LEN=2048                     # penalty starts at 6144
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# Batch shape. validate_cfg requires (policy_mini_batch_size * n_samples_per_prompt) % dp == 0;
# dp = 16/TP4 = 4 here (KDA has no context-parallel path and megatron-core rejects mHC with
# PP>1, so TP is the only divisor available). n_samples=12 rather than DAPO's usual 16.
TRAIN_BATCH_SIZE=128
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=12
EVAL_N_SAMPLES_PER_PROMPT=12
MAX_TOKENS_PER_MICROBATCH=8192  # must hold one full sequence; 16384 OOM'd at step 1 on GSM8K

# Rollout router replay (R3): vLLM returns the experts it routed each token to and Megatron
# replays that routing, so the trainer scores rollouts with the router the sampler used. On a
# 288-expert sigmoid MoE this removes most of the rollout/train logprob gap (~4x smaller
# policy/rollout_train_logprobs_abs_diff_mean). Needs distributed_executor_backend=mp (set
# below). SkyRLGymGenerator refuses R3 together with step_wise_trajectories,
# use_conversation_multi_turn=false, a custom chat_template, or vision_language_generator; this
# recipe leaves all four at R3-compatible defaults. Routing is fixed across the
# train_batch_size / policy_mini_batch_size mini-batches of a step, a small known bias.
ENABLE_ROUTING_REPLAY="${ENABLE_ROUTING_REPLAY:-false}"

# DAPO algorithm knobs (from run_megatron_dapo_qwen3.6_35b_a3b_lora.sh)
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
LR=1e-5                          # LoRA adapters, as in the reference LoRA scripts

# share_expert_adapters=False gives every expert its own adapter; with 288 experts holding ~97%
# of the parameters, one shared adapter was the capacity bottleneck. normalize_moe_lora then
# divides the expert rank by moe_router_topk (64//8 = 8), keeping the per-token expert
# contribution comparable to a dense rank-64 adapter; it requires rank % topk == 0.
LORA_RANK=64
LORA_ALPHA=64
# merge_lora=false ships a 3.9 GiB adapter in 29s against ~112s for the ~599 GiB merged path,
# vLLM then needs `experts` in lora_target_modules (see VLLM_LORA_TARGET_MODULES below).
MERGE_LORA=false
SHARE_EXPERT_ADAPTERS=false
NORMALIZE_MOE_LORA=true
LORA_TARGET_MODULES='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2,q_proj,k_proj,v_proj,b_proj,f_a_proj,g_a_proj,o_proj]'

MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1
# On 180 GiB GPUs (B200) use EP=16: at EP=8 every GPU still holds 1/8 of the experts
# (~78 GiB of frozen base) and the policy backward runs out of memory.
MEGATRON_EP=8
MEGATRON_ETP=1

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0
INFERENCE_ENGINE_MAX_NUM_SEQS=512
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.7}"

# The client fans out one request per sequence, so the router sees the whole batch at once; its
# defaults (queue_size=100, queue_timeout_secs=60) drop the overflow and the client then sees an
# empty body as "orjson.JSONDecodeError ... (char 0)". Size the queue past the batch.
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'

# vLLM-side LoRA targets, only consulted when MERGE_LORA=false. "experts" is the whole fix for
# "AssertionError: LoRA context must be set" -- supplying lora_target_modules at all flips the MoE
# from unrestricted to filtered. f_b_proj/g_b_proj stay out (KDA's non-contiguous f_a/g_a).
VLLM_LORA_TARGET_MODULES='["fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj", "gate_up_proj", "down_proj", "in_proj_qkvbfg_a", "experts"]'

if [ "$MERGE_LORA" = "false" ]; then
  LORA_ENGINE_KWARG='"lora_target_modules": '"$VLLM_LORA_TARGET_MODULES"', '
else
  LORA_ENGINE_KWARG=''
fi
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "kv_cache_dtype": "bfloat16", '"$LORA_ENGINE_KWARG"'"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "pass_config": {"fuse_allreduce_rms": false}}}'

# NCCL timeout: DP ranks finish their microbatches unevenly and the early ones sit in a
# collective. Past the 600s default torch's watchdog calls std::terminate, which surfaces on the
# driver only as a Ray ActorUnavailableError. Concurrency 128/engine: the 512 default releases
# the whole batch at once and every backend then returns 502.
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=128
export FLA_TILELANG=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# Host port 8000 is often already taken on a shared cluster; a LoadBalancer that claims it makes
# /wake_up return someone else's 404 and the run dies at the next weight sync.
export SKYRL_VLLM_START_PORT="${SKYRL_VLLM_START_PORT:-8400}"
# Otherwise redirect_actor_output_to_file() swallows vLLM's errors.
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1

RUN_NAME="${RUN_NAME:-glm5p3_flash_dapo_sync_lora_r${LORA_RANK}_8k_tp${MEGATRON_TP}}"
# Checkpoints are ~34G each (adapter + optimizer state, not the 599 GiB base).
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
  trainer.policy.language_model_only=true \
  generator.inference_engine.language_model_only=true \
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
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=64 \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.disable_parameter_transpose_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.model.lora.target_modules="$LORA_TARGET_MODULES" \
  trainer.policy.megatron_config.lora_config.merge_lora=$MERGE_LORA \
  trainer.policy.model.lora.share_expert_adapters=$SHARE_EXPERT_ADAPTERS \
  trainer.policy.megatron_config.lora_config.normalize_moe_lora=$NORMALIZE_MOE_LORA \
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
  trainer.eval_interval=25 \
  trainer.ckpt_interval=10 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.distributed_executor_backend="mp" \
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
  trainer.project_name="glm5p3_flash_dapo" \
  trainer.run_name="$RUN_NAME" \
  "$@"
