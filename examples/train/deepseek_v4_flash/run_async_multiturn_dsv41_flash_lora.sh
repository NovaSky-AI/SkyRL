set -x

# Fully async, multi-turn LoRA RL for DeepSeek-V4.1-Flash (text only) with Megatron + vLLM on separate GPUs.
#   - trainer: NUM_TRAIN_NODES x 8 GPUs, Megatron TP1 / EP (= all trainer GPUs), LoRA without merge;
#   - sampler: NUM_ENGINES vLLM engines of TP8 (one per 8-GPU node) serving the released FP8 / MXFP4 weights
#     (FP4 routed experts), the adapter synced in memory after every step;
#   - R3 (rollout routing replay), off-policy correction as masked IS (rollout_is + geometric sequence mask) and
#     score centering (needs the sampler's recorded support), multi-turn episodes with per-trajectory prefix caching.
# Defaults are 1 trainer node + 2 sampler nodes (8 + 16 GPUs). On 2 nodes: NUM_ENGINES=1.
#
# MODEL_PATH: the release in its own format, text only. Build it (all layers, minutes; or a layer slice) with
#   python examples/train/deepseek_v4_flash/make_text_only_checkpoint.py <release dir> <release dir>  (in place)
# vLLM serves it as released; Megatron dequantizes the FP8 / MXFP4 weights to BF16 on load (the frozen LoRA base).
#
# DATA_DIR / ENV_CLASS: gsm8k_multi_turn is a stand-in for a long-horizon multi-turn environment:
#   uv run --isolated --extra fsdp -m examples.train.turn_level_rewards.gsm8k_multi_turn_dataset --output_dir $DATA_DIR
# Context: MAX_TURNS x (MAX_GENERATE_LENGTH + observation) + prompt must fit MAX_MODEL_LEN (256k: MAX_MODEL_LEN=262144).
#
#   bash examples/train/deepseek_v4_flash/run_async_multiturn_dsv41_flash_lora.sh [overrides]

: "${MODEL_PATH:?text-only FP8/MXFP4 DeepSeek-V4.1-Flash dir (make_text_only_checkpoint.py)}"
: "${DATA_DIR:=$HOME/data/gsm8k_multi_turn}"
: "${ENV_CLASS:=gsm8k_multi_turn}"
: "${NUM_TRAIN_NODES:=1}"
: "${NUM_ENGINES:=2}"
: "${MAX_TURNS:=4}"
: "${MAX_PROMPT_LENGTH:=2048}"
: "${MAX_GENERATE_LENGTH:=4096}"
: "${MAX_MODEL_LEN:=32768}"
: "${TRAIN_BATCH_SIZE:=32}"          # prompts per training step
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${MAX_STALENESS_STEPS:=2}"
: "${NUM_PARALLEL_GENERATION_WORKERS:=$(( TRAIN_BATCH_SIZE * (MAX_STALENESS_STEPS + 1) ))}"
: "${MAX_TRAINING_STEPS:=20}"
: "${LOGGER:=wandb}"
: "${RUN_NAME:=dsv41_flash_async_multiturn_lora}"
: "${CKPT_PATH:=$HOME/ckpts/$RUN_NAME}"
: "${ENABLE_ROUTING_REPLAY:=true}"
: "${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:=0.85}"
# Score centering needs a bounded sampler support (top_k > 1); 20 as examples/train/top_p_replay.
: "${TOP_K:=20}"
: "${TOP_P:=1.0}"

# vLLM layout per 8-GPU engine: TP8 (default) replicates the attention KV cache on every GPU (~10M tokens per B200
# node); VLLM_DP=8 runs data-parallel attention + expert-parallel MoE (TP1 x DP8 x EP8), so each GPU caches only its
# own sessions (~13M tokens per B200 GPU, ~106M per node: 640 sessions at ~100k tokens fit). DP needs the
# SKYRL_VLLM_MOE_LORA_* EP patches below and runs only decode steps in CUDA graphs: under DP the piecewise
# (mixed prefill + decode) graphs hit an illegal memory access in vLLM 0.32 even without LoRA, so those steps run
# eagerly. 8x B200, active rank-32 adapter, 640 sessions, depth up to 200k: 99 gen tok/s/GPU at DP8.
: "${VLLM_DP:=1}"
VLLM_TP=$(( 8 / VLLM_DP ))
: "${VLLM_EP:=$( [ "$VLLM_DP" -gt 1 ] && echo 8 || echo 1 )}"  # VLLM_EP=1 with VLLM_DP>1: DP attention + TP MoE
# DP: decode steps run FULL CUDA graphs; mixed prefill + decode steps run eagerly. Under DP every rank pads to the
# largest rank's token count in graph mode (one rank's 8k prefill chunk makes all 8 ranks run 8k tokens), while eager
# steps run each rank's own tokens, so PIECEWISE graphs measured slower with desynced turns (depthfirst sim, staggered
# starts: ~735 vs ~1,100 gen tok/s/GPU). VLLM_DP_PIECEWISE_GRAPHS=1 enables them anyway; it also sets
# VLLM_USE_BREAKABLE_CUDAGRAPH=1, without which vLLM 0.32 captures DeepSeek-V4.1's attention inside the PIECEWISE
# graphs on the DP launch path (it reads the variable when the attention module is imported, before vLLM's config sets
# it) and replaying them hits an illegal memory access.
CUDAGRAPH_MODE_KWARG=""
if [ "$VLLM_DP" -gt 1 ]; then
  if [ "${VLLM_DP_PIECEWISE_GRAPHS:-0}" = 1 ]; then
    export VLLM_USE_BREAKABLE_CUDAGRAPH=1
  else
    CUDAGRAPH_MODE_KWARG=', "cudagraph_mode": "FULL_DECODE_ONLY"'
  fi
fi

LORA_RANK=32
LORA_ALPHA=32
LORA_TARGET_MODULES='[linear_q_down_proj,linear_q_up_proj,linear_kv_proj,linear_proj,linear_fc1,linear_fc2]'
VLLM_LORA_TARGET_MODULES='["fused_wqa_wkv", "wq_b", "wo_b", "gate_up_proj", "down_proj", "experts"]'
MEGATRON_EP=$(( NUM_TRAIN_NODES * 8 ))

# swa_bounded_replay: false avoids an out-of-bounds write in vLLM 0.32's LoRA kernels. With bounded replay on, the
# DSv4.1 decoder re-runs its last layers on a subset of the batch's rows in eager prefill steps
# (models/deepseek_v41/decoder_replay_layers.py), but the punica LoRA token metadata still covers the whole batch, so
# _lora_expand_kernel writes past the subset: "CUDA error: an illegal memory access" (or a hang, or NaN logprobs) at
# high concurrency, first seen ~1 min into step 1 at 32x8 rollouts. Off, the sliding-window cache joins prefix caching.
# The colocated recipe sets it off too. ASYNC_SCHEDULING=false is no longer needed (it only lowered concurrency).
: "${ASYNC_SCHEDULING:=true}"
# Chunked-prefill token budget per engine step. vLLM's default (2048) leaves ~1,400 tokens next to 640 running decodes
# while each multi-turn observation is ~1,870 tokens, so requests queue for admission and steps mix a little prefill
# into every decode (the mixed steps run piecewise CUDA graphs, launch-bound). 8x B200, TP8, LoRA on, 640 sessions
# (vLLM inference bench), gen tok/s/GPU: 2048 -> 479, 16384 -> 743, 32768 -> 898 (pure decode 576 -> 1257 /
# 1197). Costs KV cache: 78.9 -> 73.6 -> 68.0 GiB/GPU.
: "${MAX_NUM_BATCHED_TOKENS:=32768}"
: "${ENTRYPOINT:=examples.train.fully_async.main_fully_async}"
# CUDA-graph capture sizes: vLLM's default grid stops at 1024 tokens (Blackwell), but multi-turn steps mix an
# observation prefill into the running decodes (640 decodes + a chunk is ~1.7-8k tokens), and without a graph those
# steps run fully eagerly. vLLM's default grid plus 1152..8192. 8x B200, TP8, LoRA on with an active rank-32 adapter,
# 640 staggered sessions: 386 -> 462 gen tok/s/GPU; pure decode 607 -> 831; eager steps 61 -> 2 per 30 s; KV cache
# 68.0 -> 62.3 GiB/GPU.
# Tuned block sizes for vLLM's fused MoE-LoRA kernels (vllm_tuned_configs/). Without
# them a 640-token decode step pads each of 384 experts' ~10 rows to a 128-row block. 8x B200, TP8, active rank-32
# adapter, 640 staggered sessions: 462 -> 570 gen tok/s/GPU. Relative path: Ray workers run from the repo root.
# LoRA MoE on FlashInfer's TRT-LLM MXFP4 kernel instead of Marlin (patches/vllm/patch_mxfp4_trtllm_lora_moe.py):
# same MXFP4 weights and BF16 activations, different FP4 GEMM. 640 staggered sessions, active adapter: 570 -> 797.
export SKYRL_VLLM_MXFP4_TRTLLM_LORA_MOE="${SKYRL_VLLM_MXFP4_TRTLLM_LORA_MOE:-1}"
# Fused MoE-LoRA through vLLM's two-kernel (shrink + expand) path with tuned tiles instead of the one-shot kernel
# (patches/vllm/patch_moe_lora_two_kernel.py): 640 staggered sessions, active adapter, 8x B200: 756 -> 945.
export SKYRL_VLLM_MOE_LORA_TWO_KERNEL="${SKYRL_VLLM_MOE_LORA_TWO_KERNEL:-1}"
export SKYRL_VLLM_MOE_LORA_INT64_OFFSETS="${SKYRL_VLLM_MOE_LORA_INT64_OFFSETS:-1}"
export SKYRL_VLLM_MOE_LORA_EP_GLOBAL_META="${SKYRL_VLLM_MOE_LORA_EP_GLOBAL_META:-1}"
export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-examples/train/deepseek_v4_flash/vllm_tuned_configs}"
CUDAGRAPH_CAPTURE_SIZES="$(python3 -c 'print([1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 1025, 16)) + [1152, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192])')"
ENGINE_INIT_KWARGS='{"max_model_len": '"$MAX_MODEL_LEN"', "async_scheduling": '"$ASYNC_SCHEDULING"', "swa_bounded_replay": false, "lora_target_modules": '"$VLLM_LORA_TARGET_MODULES"', "max_num_batched_tokens": '"$MAX_NUM_BATCHED_TOKENS"', "compilation_config": {"cudagraph_capture_sizes": '"$CUDAGRAPH_CAPTURE_SIZES""$CUDAGRAPH_MODE_KWARG"'}}'
# consistent_hash (SkyRL's default) pins each rollout to one server via its X-Session-ID, so every turn finds its
# history in that server's prefix cache. round_robin spreads a rollout's turns over the servers: with VLLM_DP=8 (one
# server per DP rank) only 1 turn in 8 hits the cache and the rest re-prefill the whole context (~100k tokens at
# depthfirst's length; measured prefix-cache hit rate ~5%).
ROUTER_INIT_KWARGS='{"policy": "consistent_hash", "queue_size": 8192, "queue_timeout_secs": 1800}'

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=3600
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
# Megatron-Bridge's checkpoint reader keeps a safetensors handle open for every shard it touched (each one mmaps the
# whole file) and clones every tensor under 16M elements per rank, both to speed up repeated full-weight exports, which
# LoRA never does. With the full model on 1 TB Modal B300 hosts that held ~900 GB after the Megatron load and ran the
# trainer node out of memory at step 1. Off, each shard is opened, read and closed.
export MBRIDGE_STATE_REUSE_HANDLES="${MBRIDGE_STATE_REUSE_HANDLES:-0}"
export MBRIDGE_STATE_CACHE_SMALL_TENSORS="${MBRIDGE_STATE_CACHE_SMALL_TENSORS:-0}"
# Fused cuDNN dense scoring for the V4.1 indexer (Megatron-LM#7285 backport).
export SKYRL_DSV41_INDEXER_IMPL="${SKYRL_DSV41_INDEXER_IMPL:-fused}"

uv run --isolated --extra megatron -m "$ENTRYPOINT" \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=$ENV_CLASS \
  generator.max_turns=$MAX_TURNS \
  generator.batched=false \
  `# vLLM's V4.1 decode drifts after a prefix-cache hit from another trajectory (#2573 HANDOFF); a per-trajectory` \
  `# salt keeps each episode's own turns cached, which is where multi-turn's hit rate comes from.` \
  generator.cache_salt_per_trajectory=true \
  trainer.fully_async.enabled=true \
  trainer.fully_async.max_staleness_steps=$MAX_STALENESS_STEPS \
  trainer.fully_async.num_parallel_generation_workers=$NUM_PARALLEL_GENERATION_WORKERS \
  trainer.fully_async.clear_kv_cache_on_weight_sync=false \
  trainer.strategy=megatron \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=$NUM_TRAIN_NODES \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.policy.megatron_config.tensor_model_parallel_size=1 \
  trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
  trainer.policy.megatron_config.context_parallel_size=1 \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
  trainer.policy.megatron_config.mtp_num_layers=0 \
  trainer.policy.megatron_config.moe_grouped_gemm=true \
  trainer.policy.megatron_config.moe_token_dispatcher_type="alltoall" \
  trainer.policy.megatron_config.moe_enable_routing_replay=$ENABLE_ROUTING_REPLAY \
  generator.inference_engine.enable_return_routed_experts=$ENABLE_ROUTING_REPLAY \
  `# As the 2-node LoRA recipe: DSv4 attention rejects core_attn recompute, mHC rejects full recompute.` \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="selective" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[moe,mhc,layernorm] \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.model.lora.target_modules="$LORA_TARGET_MODULES" \
  trainer.policy.model.lora.sync_mode=memory \
  trainer.policy.megatron_config.lora_config.merge_lora=false \
  `# Masked IS against the sampler (MIS) + score centering; both optimize against the rollout logprobs, so the` \
  `# separate old-logprob forward is skipped.` \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="rollout_is" \
  trainer.algorithm.off_policy_correction.sequence_mask_metric=geometric \
  trainer.algorithm.off_policy_correction.geo_mask_high=1.01 \
  trainer.algorithm.off_policy_correction.geo_mask_low=0.99 \
  trainer.algorithm.enable_score_centering=true \
  trainer.algorithm.enable_sample_support_replay=true \
  generator.inference_engine.enable_return_sample_support_set=true \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.loss_reduction="token_mean" \
  trainer.policy.optimizer_config.lr=1e-5 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  `# Fused LM-head logprob (Triton row kernels); no packed (THD) sequences for DSv4 attention.` \
  trainer.fused_lm_head_logprob=true \
  trainer.fused_lm_head_logprob_backend=triton \
  trainer.remove_microbatch_padding=false \
  trainer.use_expandable_segments=true \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=1 \
  trainer.max_training_steps=$MAX_TRAINING_STEPS \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=10 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$VLLM_TP \
  generator.inference_engine.data_parallel_size=$VLLM_DP \
  generator.inference_engine.expert_parallel_size=$VLLM_EP \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=$INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.router_init_kwargs="$ROUTER_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.top_k=$TOP_K \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  trainer.logger="$LOGGER" \
  trainer.project_name="dsv41_flash_async" \
  trainer.run_name="$RUN_NAME" \
  "$@"
