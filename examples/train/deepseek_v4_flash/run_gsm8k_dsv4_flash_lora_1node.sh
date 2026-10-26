set -x

# Colocated GRPO training+generation for DeepSeek-V4-Flash on GSM8K with Megatron + LoRA.
# Single node of 8xB200 (178 GiB/GPU).
#
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   export WANDB_API_KEY=<your_key_here>   # or leave LOGGER=console
#   bash examples/train/deepseek_v4_flash/run_gsm8k_dsv4_flash_lora_1node.sh
#
# Short-context smoke run: prompts capped at 512 tokens, responses at 1024, a 2k engine context.
#
# Why LoRA: DeepSeek-V4-Flash is ~284B params (~530 GiB bf16, 97% of it routed experts). A full
# fine-tune needs ~16 bytes/param of training state (~4.6 TiB) against 1.4 TiB of HBM on one node.
# LoRA keeps the frozen base at ~86 GiB/GPU (attention and mHC replicated at TP=1, experts at EP=8)
# and puts the only trainable state in the adapters.
#
# MODEL_PATH is a BF16 copy, not the release: deepseek-ai/DeepSeek-V4-Flash ships FP8 attention and
# MXFP4 experts. Megatron-Bridge dequantizes those on import, but vLLM must serve the same BF16
# weights the trainer syncs into it. Build the copy with
#   DSV4_LOCAL_DIR=<release dir> python examples/train/deepseek_v4_flash/dev/make_slice.py <out> $(seq -s, 0 42)
# (every layer; MTP dropped; chat template embedded -- the release ships none).

MODEL_PATH="${MODEL_PATH:-$HOME/models/DeepSeek-V4-Flash-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="${LOGGER:-console}"  # change to "wandb" to log to wandb

# Stage 1 smoke run: an epoch is ~233 steps, and a step is ~16 min (training is kernel-launch-bound:
# small padded microbatches through mHC's Sinkhorn iterations, compressor and indexer per layer).
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-10}"

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

# 32 prompts x 8 samples = 256 sequences per step, one optimizer step per batch. DP is 8 (TP=1).
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
# Unpacked microbatches (see remove_microbatch_padding below): each holds this many padded rows.
MICRO_TRAIN_BATCH_SIZE_PER_GPU=2
MICRO_FORWARD_BATCH_SIZE_PER_GPU=4

USE_KL_LOSS=false   # no ref model is constructed when this (and use_kl_in_reward) are false
LR=1e-5             # LoRA: higher LR for adapters than for full FT

# LoRA. target_modules must be spelled out: "all-linear" maps to dense-attention names
# (linear_qkv/...) that match none of DSv4's projections. mcore names from DSv4HybridAttention
# (wq_a -> linear_q_down_proj, wq_b -> linear_q_up_proj, wkv -> linear_kv_proj, wo_b ->
# linear_proj) plus the routed/shared-expert MLP. wo_a (a grouped raw parameter), the KV
# compressor and the lightning indexer carry no adapter.
LORA_RANK=32
LORA_ALPHA=32
# merge_lora=false syncs only the adapter (the path the Stage 0 LoRA roundtrip row validates);
# it needs `experts` in vLLM's lora_target_modules -- supplying the list at all switches vLLM's MoE
# LoRA wrapping from unrestricted to filtered, and an unwrapped MoE layer fails the profile run.
MERGE_LORA=false
LORA_TARGET_MODULES='[linear_q_down_proj,linear_q_up_proj,linear_kv_proj,linear_proj,linear_fc1,linear_fc2]'
# vLLM names for the same modules: wq_a and wkv are packed into fused_wqa_wkv, w1/w3 into gate_up_proj.
VLLM_LORA_TARGET_MODULES='["fused_wqa_wkv", "wq_b", "wo_b", "gate_up_proj", "down_proj", "experts"]'

# Megatron mesh. megatron-core's DSv4 hybrid attention is TP=1-only at the pinned rev, so attention
# is replicated and EP shards the experts (32/GPU). PP stays 1 (hash-routed layers must sit on the
# embedding's stage; mHC + PP is untested here).
MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Rollout router replay (R3). Off for this smoke run; both knobs move together.
ENABLE_ROUTING_REPLAY=false

# vLLM's share of each GPU: 0.6 -> ~107 GiB, ~66 GiB of it the BF16 shard at TP8. Fits next to
# Megatron's ~86 GiB frozen base only because colocate_all sleeps vLLM during training.
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.6}"

if [ "$MERGE_LORA" = "false" ]; then
  LORA_ENGINE_KWARG='"lora_target_modules": '"$VLLM_LORA_TARGET_MODULES"', '
else
  LORA_ENGINE_KWARG=''
fi
# Leave prefix caching on: R3 (enable_return_routed_experts) needs it for vLLM's AuxOutput connector.
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', '"${LORA_ENGINE_KWARG%, }"'}'

# TileLang JITs vLLM's mHC kernels with CUDA_HOME's nvcc; without it on PATH it falls back to the
# venv's pip nvcc, which rejects the venv's CCCL headers. SkyRL forwards CUDA_HOME to the actors.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
# GCP A4 login shells set Google's gIB NCCL net plugin, built for an older NCCL than the venv's;
# NCCL's built-in IB transport over the RoCE NICs works. No-op elsewhere.
unset NCCL_NET NCCL_TUNER_CONFIG_PATH
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# Fresh FlashInfer autotune cache per run: with a warm one, vLLM boot hung in the collective MoE tuning
# (rank 0 loads the cache, the others wait) until the health timeout. Each node creates its own empty dir.
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-${TMPDIR:-/tmp}/vllm_autotune_$(date +%s)}"
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1800

RUN_NAME="dsv4_flash_gsm8k_lora_r${LORA_RANK}_tp${MEGATRON_TP}_ep${MEGATRON_EP}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.loss_reduction="sequence_mean" \
  trainer.policy.model.path="$MODEL_PATH" \
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
  trainer.policy.megatron_config.moe_enable_routing_replay=$ENABLE_ROUTING_REPLAY \
  generator.inference_engine.enable_return_routed_experts=$ENABLE_ROUTING_REPLAY \
  `# No recompute: megatron-core's DSv4 attention rejects core_attn checkpointing, and the worker` \
  `# drops it (leaving recompute off). Activation memory is bounded by the short context here.` \
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
  `# Per-user adapter dir (the default /tmp/skyrl_lora_sync can belong to another user on a shared node). With` \
  `# experts adapted the adapter is ~20 GB per sync, so point LORA_SYNC_PATH at big local storage on every node.` \
  trainer.policy.model.lora.lora_sync_path="${LORA_SYNC_PATH:-${TMPDIR:-/tmp}/skyrl_lora_sync_${USER}}" \
  trainer.policy.megatron_config.lora_config.merge_lora=$MERGE_LORA \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  `# megatron-core's DSv4 compressed attention takes no packed (THD) sequences at the pinned rev;` \
  `# the unpacked path left-aligns rows and pads to the largest compress ratio.` \
  trainer.remove_microbatch_padding=false \
  trainer.use_expandable_segments=true \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BATCH_SIZE_PER_GPU \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE_PER_GPU \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=1 \
  trainer.max_training_steps=$MAX_TRAINING_STEPS \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.ckpt_interval=-1 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/$RUN_NAME" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.gpu_memory_utilization=$INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name="dsv4_flash_gsm8k" \
  trainer.run_name="$RUN_NAME" \
  "$@"
