set -x

# Colocated GRPO training+generation for GLM-5.3-Flash on GSM8K with Megatron + LoRA.
# Single node of 8xB300 (275 GiB/GPU).
#
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   export WANDB_API_KEY=<your_key_here>   # or leave LOGGER=console
#   bash examples/train/glm5_3_flash/run_gsm8k_glm5p3_flash_lora_1node.sh
#
# Short-context smoke run: prompts are capped at 512 tokens and responses at 2048, so the
# engine only ever needs a 4k context and rollouts stay cheap. Bump MAX_RESPONSE_LENGTH if
# you want the model to actually finish its reasoning on the harder GSM8K items.
#
# Why LoRA: GLM-5.3-Flash is ~313B params (~599 GiB bf16, 97% of it routed experts) with ~17B
# activated. A full fine-tune needs bf16 weights + bf16 grads + fp32 master/m/v, i.e. ~5 TiB of
# state against 2.2 TiB of HBM on one node, and the expert shards get no optimizer sharding at
# EP8/ETP1 on 8 ranks (expert-DP == 1). LoRA keeps the frozen base at ~75 GiB/GPU and puts the
# only trainable state in the adapters.
#
# MODEL_PATH is the BF16 release, not the default fp8 one. zai-org/GLM-5.3-Flash ships e4m3
# weights with blockwise *.weight_scale_inv sidecars, and Glm5NextBridge does not override
# maybe_modify_loaded_hf_weight -- the import would cast fp8 -> bf16 and silently drop the block
# scales (see DeepSeekV3Bridge for the override that handles this). BF16 sidesteps that entirely.

MODEL_PATH="${MODEL_PATH:-/data/trajectory/model-cache/glm5p3-flash-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="${LOGGER:-console}"  # change to "wandb" to log to wandb

# Short context: GSM8K prompts are tiny, and a 2k response cap keeps the rollout phase short.
# (Sequences past dsa_indexer_topk=2048 are fine on their own -- megatron-core's k-pool indexer
# handles them -- this cap is just to keep the smoke run cheap.)
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

# Small on-policy batch for a smoke run: 32 prompts x 8 samples = 256 sequences per step,
# one optimizer step per batch. Dense DP is 8 / (TP2 * PP1 * CP1) = 4, so 64 seqs per DP rank.
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
# Peak training memory is dominated by this, not by batch size (micro_*_batch_size_per_gpu=1
# means one packed microbatch of this many tokens). At 16384 step 1 OOM'd trying to add 31.6 GiB
# on top of ~230 GiB in use, against ~75 GiB of frozen LoRA base weights per GPU.
MAX_TOKENS_PER_MICROBATCH=4096

USE_KL_LOSS=false   # no ref model is constructed when this (and use_kl_in_reward) are false
LR=1e-5             # LoRA: higher LR for adapters than the 1e-6 used for full-FT

# LoRA config. target_modules must be spelled out: the "all-linear" default maps to the
# dense-attention names (linear_qkv/...), which match none of GLM-5.3-Flash's MLA or KDA
# projections. These are the mcore module names from glm5_next/bridge.py (MLA) and
# mcore_ext/kda.py (KDA), plus the MoE/dense MLP linears.
LORA_RANK=32
LORA_ALPHA=32
# merge_lora=true keeps this smoke run on the simple path (vLLM runs without LoRA). false syncs
# only the adapter (a few GiB instead of the ~599 GiB merged model) and is what the DAPO recipe
# uses; it needs `experts` in vLLM's lora_target_modules, because supplying that list at all
# switches vLLM's MoE LoRA wrapping from unrestricted to filtered, and an unwrapped MoE layer
# fails vLLM's profile run with "AssertionError: LoRA context must be set". MERGE_LORA is a
# plain assignment on purpose: VLLM_LORA_TARGET_MODULES gating below keys off this shell
# variable, so flip it here rather than with a CLI override. Leave vLLM's
# enable_moe_shared_loras at its default (False): the bridge exports per-expert adapters, which
# is the layout vLLM's default MoE LoRA path consumes. language_model_only only zeroes the
# multimodal limits, so vLLM still builds Glm5NextForConditionalGeneration (SupportsLoRA).
MERGE_LORA=true
# f_b_proj / g_b_proj are deliberately absent: vLLM's KDA runs one fused GEMM (in_proj_qkvbfg_a)
# and .split()s it, so f_a/g_a are non-contiguous views and a LoRA-wrapped f_b_proj(f_a) trips
# `assert inputs.is_contiguous()` in the triton lora_shrink. Every other KDA projection is
# adapted. They must be excluded on both sides -- see VLLM_LORA_TARGET_MODULES below.
LORA_TARGET_MODULES='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2,q_proj,k_proj,v_proj,b_proj,f_a_proj,g_a_proj,o_proj]'

# Megatron mesh. EP is the scaling dimension for a MoE this sparse (36 experts/GPU); TP only
# has to cover the ~9B of non-expert weights. PP stays at 1 because megatron-core rejects mHC
# with pipeline_model_parallel_size > 1, and CP at 1 because KDA has no context-parallel path.
MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

# Adapter optimizer state is small, but CPU offload is what the multi-node GLM-5.3-Flash runs
# use and it costs little here. use_precision_aware_optimizer stays off (it can break ckpt save).
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Rollout router replay (R3): vLLM returns the experts it routed to and Megatron replays that
# routing, keeping rollout/train logprobs from drifting on a 288-expert MoE. Off for this smoke
# run; both knobs move together (validate_cfg rejects replay without the other). R3 works with
# merge_lora=false.
ENABLE_ROUTING_REPLAY=false

# GLM-5.3-Flash is shipped as a VL checkpoint; SkyRL bridges only the language model, and the
# KDA layers need packed (thd) sequences.
LANGUAGE_MODEL_ONLY=true

# vLLM's KDA triton kernels put (num_seqs * kda_heads) in CUDA grid dim y; the default
# max_num_seqs=1024 with 64 heads exceeds the 65535 limit and CUDA-graph capture fails.
INFERENCE_ENGINE_MAX_NUM_SEQS=512

# vLLM's share of each 268.6 GiB B300: 0.7 -> ~188 GiB, of which ~75 GiB is the BF16 weight
# shard at TP8, leaving ~113 GiB of KV pool. That sits on top of Megatron's ~75 GiB frozen
# base, so it only fits because colocate_all sleeps vLLM during the training phase.
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.7}"

# vLLM-side names for the same set as LORA_TARGET_MODULES: KDA's q/k/v/b/f_a/g_a fuse into
# in_proj_qkvbfg_a, MLA's q_a/kv_a into fused_qkv_a_proj. This list controls which modules vLLM
# *wraps* (it is not inferred from the adapter), so f_b_proj/g_b_proj must be excluded here too.
# "experts" is required: setting lora_target_modules at all switches the MoE from
# "unrestricted" to "filtered", and its module suffix is `experts`. vLLM picks a LoRA-aware
# MoE expert kernel whenever LoRA is enabled globally (oracle/unquantized.py:222) but only
# sets the lora_context that kernel asserts on when the MoE layer is itself wrapped -- so
# omitting `experts` here is what produced "LoRA context must be set". Only consulted when
# MERGE_LORA=false.
VLLM_LORA_TARGET_MODULES='["fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj", "gate_up_proj", "down_proj", "in_proj_qkvbfg_a", "experts"]'

if [ "$MERGE_LORA" = "false" ]; then
  LORA_ENGINE_KWARG='"lora_target_modules": '"$VLLM_LORA_TARGET_MODULES"', '
else
  LORA_ENGINE_KWARG=''
fi
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "kv_cache_dtype": "bfloat16", '"$LORA_ENGINE_KWARG"'"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "pass_config": {"fuse_allreduce_rms": false}}}'

# fla's TileLang backend aborts on Blackwell; KDA runs fla kernels, so force Triton.
export FLA_TILELANG=0
# TileLang still JITs other kernels (vLLM's fused mHC path), and it takes its toolkit from
# CUDA_HOME -- defaulting to the pip wheel tree, where nvidia-cuda-nvcc==13.3 sits next to
# nvidia-cuda-runtime==13.0 and nvidia-cuda-cccl rejects the pair ("CUDA compiler and CUDA
# toolkit headers are incompatible"). Point it at the self-consistent system toolkit; SkyRL
# forwards CUDA_HOME into the Ray runtime env so the engine/trainer actors see it too.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

RUN_NAME="glm5p3_flash_gsm8k_lora_r${LORA_RANK}_tp${MEGATRON_TP}_ep${MEGATRON_EP}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.loss_reduction="sequence_mean" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=1 \
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
  `# mHC is not compatible with full activation recompute in this megatron-core ("enable_mhc_connections` \
  `# is not yet compatible with full activation recompute"), so this must be selective with 'mhc' in` \
  `# recompute_modules -- but SkyRL's HyperConnectionTransformerLayer then rejects 'mhc' itself (it` \
  `# does not thread mcore's CheckpointWithoutOutputManager, so accepting it would silently drop the` \
  `# recompute), and also rejects 'layernorm'/'mlp'. That leaves core_attn + moe. Selective` \
  `# additionally requires recompute_num_layers to be None. 'gdn' is not a valid choice either` \
  `# (megatron-core has gdp_qkv/gdn_norm_out/gdp_in_proj instead, each gated on an attention variant).` \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="selective" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[core_attn,moe] \
  `# DEFAULT_TRANSFORMER_CONFIG_KWARGS injects uniform/1, which selective rejects -- null them.` \
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
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
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
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
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
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm5p3_flash_gsm8k" \
  trainer.run_name="$RUN_NAME" \
  "$@"
