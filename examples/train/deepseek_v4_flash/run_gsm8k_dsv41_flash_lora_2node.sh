set -x

# Colocated GRPO training+generation for DeepSeek-V4.1-Flash (text only) on GSM8K with Megatron + LoRA.
# 2 nodes x 8xB200 (178 GiB/GPU) against a running Ray cluster (`ray start` on both nodes with the uv env's
# Ray version; MODEL_PATH and DATA_DIR present on every node).
#
#   export WANDB_API_KEY=<key>   # or leave LOGGER=console
#   bash examples/train/deepseek_v4_flash/run_gsm8k_dsv41_flash_lora_2node.sh
#
# Short-context Stage 1 smoke run: prompts capped at 512 tokens, responses at 1024, a 2k engine context.
#
# MODEL_PATH is a text-only BF16 copy of deepseek-ai/DeepSeek-V4.1-Flash-1017 with every layer:
#   python examples/train/deepseek_v4_flash/dev/make_slice_v41.py <out> $(seq -s, 0 39)
# (FP8 attention / MXFP4 experts dequantized, vision + DSpark MTP dropped, Engram tables kept FP8.)
#
# Memory. ~551B params (543B of them routed experts, ~1.1 TB bf16) plus two frozen ~101 GB FP8 Engram tables.
# - Megatron at EP=16: ~68 GiB experts + ~16 GiB replicated attention/mHC/embeddings + ~25 GiB of Engram rows
#   (each table row-sharded over the node's 8 GPUs) per GPU; only the LoRA adapters train.
# - vLLM: at TP=8 one engine would hold ~137 GiB of weights per GPU, so one engine spans both nodes as TP=8 x
#   PP=2 (~68 GiB per GPU). The PP boundary must not split a V4.1 KV-/index-/candidate-sharing group, and
#   vLLM's even split of 40 layers (0-19 | 20-39) is the only balanced cut that does not. vLLM keeps its Engram
#   tables in pinned host memory.
# colocate_all sleeps vLLM during training and offloads the policy during rollout.

MODEL_PATH="${MODEL_PATH:-$HOME/models/DeepSeek-V4.1-Flash-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
NUM_NODES=2
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
INFERENCE_ENGINE_PIPELINE_PARALLEL_SIZE=2
LOGGER="${LOGGER:-console}"

MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-10}"

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

# 32 prompts x 8 samples = 256 sequences per step, one optimizer step per batch. DP is 16 (TP=1).
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
MICRO_TRAIN_BATCH_SIZE_PER_GPU=1
MICRO_FORWARD_BATCH_SIZE_PER_GPU=2

USE_KL_LOSS=false
LR=1e-5

# Same module lists as DeepSeek-V4 (the Stage 0 V4.1 LoRA row validates them). The compressor, indexer and
# Engram projections carry no adapter (vLLM's ratio-1 compressor fused_wkv_wgate has a single slice).
LORA_RANK=32
LORA_ALPHA=32
MERGE_LORA=false
LORA_TARGET_MODULES='[linear_q_down_proj,linear_q_up_proj,linear_kv_proj,linear_proj,linear_fc1,linear_fc2]'
VLLM_LORA_TARGET_MODULES='["fused_wqa_wkv", "wq_b", "wo_b", "gate_up_proj", "down_proj", "experts"]'

MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=16
MEGATRON_ETP=1

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

ENABLE_ROUTING_REPLAY=false

INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.6}"
# vLLM's SM100 default V4.1 attention is FP8-only (and permutes wq_b / wo_a once at load, which an in-place
# weight update would leave unpermuted); BF16 weights need FlashMLA sparse.
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "attention_backend": "FLASHMLA_SPARSE_DSV41", "lora_target_modules": '"$VLLM_LORA_TARGET_MODULES"'}'

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
unset NCCL_NET NCCL_TUNER_CONFIG_PATH
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-${TMPDIR:-/tmp}/vllm_autotune_$(date +%s)}"
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=3600

RUN_NAME="${RUN_NAME:-dsv41_flash_gsm8k_lora_r${LORA_RANK}_ep${MEGATRON_EP}}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.loss_reduction="sequence_mean" \
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
  `# Selective recompute as in the DSv4 recipes (mhc/layernorm under LoRA rely on` \
  `# patch_checkpoint_without_output_frozen_inputs).` \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="selective" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[moe,mhc,layernorm] \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.disable_parameter_transpose_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  `# the full CPU optimizer offload turns this off by default; without it NCCL can OOM lazily allocating buffers.` \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.model.lora.target_modules="$LORA_TARGET_MODULES" \
  trainer.policy.model.lora.lora_sync_path="${LORA_SYNC_PATH:-${TMPDIR:-/tmp}/skyrl_lora_sync_${USER}}" \
  trainer.policy.megatron_config.lora_config.merge_lora=$MERGE_LORA \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
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
  trainer.ckpt_path="${CKPT_PATH:-$HOME/ckpts/$RUN_NAME}" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.pipeline_parallel_size=$INFERENCE_ENGINE_PIPELINE_PARALLEL_SIZE \
  generator.inference_engine.gpu_memory_utilization=$INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name="dsv41_flash_gsm8k" \
  trainer.run_name="$RUN_NAME" \
  "$@"
