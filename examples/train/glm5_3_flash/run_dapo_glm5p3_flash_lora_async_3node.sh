set -x

# Fully-async DAPO training for GLM-5.3-Flash with Megatron + LoRA, NON-colocated:
# 2 nodes (16 GPUs) train, 1 node (8 GPUs) serves inference. Stops after MAX_TRAINING_STEPS.
#
#   bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   export WANDB_API_KEY=<key>
#   bash examples/train/glm5_3_flash/run_dapo_glm5p3_flash_lora_async_3node.sh
#
# Sized to match run_dapo_glm5p3_flash_lora_sync_3node.sh in samples seen:
# sync does 50 steps x 128 prompts; this does 200 steps x 32 prompts = the same 6400 prompts
# (and with the same n_samples_per_prompt, the same 76800 sequences).

MODEL_PATH="${MODEL_PATH:-/data/trajectory/model-cache/glm5p3-flash-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"

# NOTE: sized for 2 nodes. vmnode-6r3vaf61zkut was drained from the Ray cluster after its GPU0
# lost P2P with every peer (nvidia-smi topo -p2p r shows NS across GPU0's row), which made vLLM's
# TP=8 ncclCommInitRank fail on that node every time. It cannot be reset in-guest -- the GPUs are
# passthrough -- and its NVLinks have been inactive since its fabric manager died on 2026-07-18.
# With only two nodes the policy drops to one node so the inference engine still gets a whole
# node to itself; restore NUM_NODES=2 for a 3-node cluster.
NUM_NODES=1                      # trainer node; the other node hosts the inference engine
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="${LOGGER:-wandb}"

MAX_TRAINING_STEPS=200

# Sequence budget. The DSA layers index with dsa_indexer_topk=2048, and megatron-core now
# implements the k-pool indexer (NVIDIA/Megatron-LM#7054), so sequences past that are selected
# rather than refused -- the earlier 896+1024 cap is gone. This is still short of the stock DAPO
# recipe's 2k prompt + 8k response: an 8k response is ~8x the per-step cost of the 1024 runs that
# measured ~10 min/step, which does not fit an overnight experiment. Overlong filtering absorbs
# the truncated tail.
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
INFERENCE_ENGINE_MAX_MODEL_LEN=6656          # prompt + response + headroom for chat-template tokens
OVERLONG_BUFFER_LEN=1024                     # penalty starts at 3072; see the note below
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# dp here is 16/TP2 = 8, so (mini * n_samples) % dp == 0 holds for any n_samples; 12 is kept
# only to match the sync run's sequences-per-prompt.
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=12
EVAL_N_SAMPLES_PER_PROMPT=12
MAX_TOKENS_PER_MICROBATCH=8192  # must hold one full sequence; 16384 OOM'd at step 1 on GSM8K

# Fully-async knobs
MAX_STALENESS_STEPS=4
NUM_PARALLEL_GENERATION_WORKERS=$(( MINI_BATCH_SIZE * (MAX_STALENESS_STEPS + 1) ))
SEQUENCE_MASK_METRIC=geometric
GEO_MASK_HIGH=1.01
GEO_MASK_LOW=0.99

# DAPO algorithm knobs. policy_loss_type=rollout_is (not dual_clip) because generation runs
# off-policy here; the geometric sequence mask above is the off-policy correction.
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

# The previous 45-step run at rank 32 / shared expert adapters moved implied accuracy only
# 0.482 -> 0.502 while reward tracked length at corr -0.93. Two changes to that:
#
# share_expert_adapters=False gives every expert its own adapter instead of one shared across all
# local grouped experts -- with 288 experts holding ~97% of the parameters, the shared adapter was
# the capacity bottleneck. normalize_moe_lora then divides the expert rank by moe_router_topk
# (8 here, so expert rank 64//8 = 8), keeping the per-token expert contribution comparable to a
# dense rank-64 adapter; it requires rank % topk == 0, which 64 satisfies.
LORA_RANK=64
LORA_ALPHA=64
MERGE_LORA=true
SHARE_EXPERT_ADAPTERS=false
NORMALIZE_MOE_LORA=true
LORA_TARGET_MODULES='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2,q_proj,k_proj,v_proj,b_proj,f_a_proj,g_a_proj,o_proj]'

MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0
INFERENCE_ENGINE_MAX_NUM_SEQS=512
# The inference node is not shared with a policy shard here, so vLLM can take more of it.
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.85}"

# The client fans out one HTTP request per sequence (batch x n_samples) and throttles only at
# SKYRL_GENERATE_CONCURRENCY_PER_ENGINE (512) x num_engines, so the router sees the whole batch at
# once. Its defaults -- queue_size=100, queue_timeout_secs=60 -- then drop the overflow, and the
# client gets an empty body: "orjson.JSONDecodeError: unexpected character ... (char 0)". Size the
# queue past the batch and give it the same deadline as request_timeout_secs. round_robin spreads
# the load across engines (and drops consistent_hash's very chatty per-request debug logging).
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'

ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "kv_cache_dtype": "bfloat16", "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "pass_config": {"fuse_allreduce_rms": false}}}'

# The client fires one HTTP request per sequence and caps in-flight work at
# SKYRL_GENERATE_CONCURRENCY_PER_ENGINE x num_engines. At the 512 default that is 1536 requests
# released at once for the sync shape, and every backend then returns
#   502 "Backend request failed: error sending request for url ..."
# uniformly (all three engines failed in equal measure), which surfaces client-side only as
# "orjson.JSONDecodeError ... (char 0)". This is the mitigation the env var documents.
# The working GSM8K run peaked at ~256 in flight against one engine, so 128/engine is well inside
# what the routers and uvicorn accept queues handled there.
# DAPO steps are far heavier than the GSM8K ones (responses run to the full cap instead of
# ~250 tokens), so a single fwd_logprobs pass took 37 min. DP ranks finish their microbatches
# unevenly, and the ones that finish early then sit in a collective: past the 600s default here,
# torch's NCCL watchdog calls std::terminate and the worker dies with
#   c10d::ProcessGroupNCCL::Watchdog::run() -> SIGABRT / "Fatal Python error: Aborted",
# which surfaces on the driver only as a Ray ActorUnavailableError (keepalive watchdog timeout).
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=128
export FLA_TILELANG=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

RUN_NAME="glm5p3_flash_dapo_async_lora_r${LORA_RANK}_2train1gen"

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo_fully_async \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.strategy=megatron \
  trainer.fully_async.enabled=true \
  trainer.fully_async.max_staleness_steps=$MAX_STALENESS_STEPS \
  trainer.fully_async.num_parallel_generation_workers=$NUM_PARALLEL_GENERATION_WORKERS \
  trainer.fully_async.clear_kv_cache_on_weight_sync=false \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="rollout_is" \
  trainer.algorithm.off_policy_correction.sequence_mask_metric=$SEQUENCE_MASK_METRIC \
  trainer.algorithm.off_policy_correction.geo_mask_high=$GEO_MASK_HIGH \
  trainer.algorithm.off_policy_correction.geo_mask_low=$GEO_MASK_LOW \
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
  trainer.placement.colocate_all=false \
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
  trainer.policy.megatron_config.moe_enable_routing_replay=false \
  generator.inference_engine.enable_return_routed_experts=false \
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
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.ckpt_interval=-1 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/$RUN_NAME" \
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
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm5p3_flash_dapo" \
  trainer.run_name="$RUN_NAME" \
  "$@"
