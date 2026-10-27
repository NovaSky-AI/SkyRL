set -x

# Colocated GRPO training+generation for DeepSeek-V4.1-Flash (text only) on GSM8K with Megatron, full fine-tuning.
# 6 nodes x 8xB200 (178 GiB/GPU), all colocated, against a running Ray cluster (`ray start` on every node with the
# uv env's Ray version; MODEL_PATH and DATA_DIR present on every node).
#
#   export WANDB_API_KEY=<key>   # or leave LOGGER=console
#   bash examples/train/deepseek_v4_flash/run_gsm8k_dsv41_flash_fullft_6node.sh
#
# Memory. ~551B params (543B routed experts). megatron-core's distributed optimizer keeps fp32 main params on GPU even
# with optimizer_cpu_offload, and with EP == DP the expert main params are not sharded. EP must divide 384 experts, so
# 6 nodes (EP=48, 8 experts/GPU) is the smallest layout on these nodes. Per GPU: ~23 GiB bf16 experts + ~45 GiB fp32
# expert main params + ~23 GiB bf16 expert grads, ~16 GiB replicated attention/mHC/embeddings + ~16 GiB of their
# grads, plus activations. The frozen Engram tables live in pinned host memory (SKYRL_DSV41_ENGRAM_TABLE_DEVICE=cpu)
# instead of ~25 GiB/GPU of HBM. vLLM: three engines of TP8 x PP2 (one per node pair; see
# run_gsm8k_dsv41_flash_lora_2node.sh for why PP2), sleeping (level 2) during training.
#
# MODEL_PATH is the text-only BF16 copy with the chat-mode template as the tokenizer's own (see
# run_dapo_dsv41_flash_lora_sync_2node.sh for how to build DeepSeek-V4.1-Flash-bf16-chat).

MODEL_PATH="${MODEL_PATH:-$HOME/models/DeepSeek-V4.1-Flash-bf16-chat}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
NUM_NODES=6
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=3          # one engine per node pair (TP8 x PP2)
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
INFERENCE_ENGINE_PIPELINE_PARALLEL_SIZE=2
LOGGER="${LOGGER:-console}"  # change to "wandb" to log to wandb

# Stage 1 smoke run.
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-10}"

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

# 48 prompts x 8 samples = 384 sequences per step, one optimizer step per batch. DP is 48 (TP=1).
TRAIN_BATCH_SIZE=48
MINI_BATCH_SIZE=48
N_SAMPLES_PER_PROMPT=8
# Unpacked microbatches (see remove_microbatch_padding below): each holds this many padded rows.
MICRO_TRAIN_BATCH_SIZE_PER_GPU=1
MICRO_FORWARD_BATCH_SIZE_PER_GPU=4

USE_KL_LOSS=false   # no ref model is constructed when this (and use_kl_in_reward) are false
LR=1e-6

# Megatron mesh. megatron-core's DSv4 hybrid attention is TP=1-only at the pinned rev, so attention is replicated
# and EP shards the experts (8/GPU).
MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=48
MEGATRON_ETP=1

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Rollout router replay (R3). Off for this smoke run; both knobs move together.
ENABLE_ROUTING_REPLAY=false

# vLLM's share of each GPU: 0.6 -> ~107 GiB, ~68 GiB of it the BF16 shard at TP8 x PP2.
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.6}"

# BF16 V4.1 needs vLLM's FlashMLA sparse backend (the SM100 default is FP8-only and permutes wq_b / wo_a once at
# load, which an in-place weight sync would leave unpermuted).
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "enable_prefix_caching": false, "attention_backend": "FLASHMLA_SPARSE_DSV41"}'

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
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=3600
export SKYRL_DSV41_ENGRAM_TABLE_DEVICE=cpu
# `kill -USR1 <worker pid>` dumps Python stacks to the worker log (diagnosing collective hangs).
export SKYRL_DSV41_FAULTHANDLER=${SKYRL_DSV41_FAULTHANDLER:-1}

RUN_NAME="${RUN_NAME:-dsv41_flash_gsm8k_fullft_ep${MEGATRON_EP}}"

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
  `# Selective recompute: megatron-core's DSv4 attention rejects core_attn checkpointing and mHC` \
  `# rejects full recompute. mHC keeps 4 residual streams per sub-layer, so recompute it with the MoE` \
  `# layers (megatron-core forbids mhc together with mlp).` \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="selective" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[moe,mhc,layernorm] \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null \
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=64 \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.disable_parameter_transpose_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  `# the full CPU optimizer offload turns this off by default; without it NCCL can OOM lazily allocating buffers.` \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  `# Precision-aware: the optimizer reads the bf16 grads directly. Off, megatron-core first materializes an fp32` \
  `# copy of every grad on GPU (~45 GiB of expert grads at EP=48, since EP == DP leaves experts unsharded) and` \
  `# OOMs in optim_step. The GLM-5.3 recipes keep it off because it can break checkpoint saving.` \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=${PRECISION_AWARE_OPTIMIZER:-true} \
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=false \
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
  `# vLLM's V4.1 decode drifts from prefill after a prefix-cache hit (logprob outliers clustered by prompt); R3 needs` \
  `# prefix caching on, so give each sample its own cache salt instead (~1% rollout time); see GeneratorConfig.cache_salt_per_trajectory.` \
  generator.cache_salt_per_trajectory=true \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name="dsv41_flash_gsm8k" \
  trainer.run_name="$RUN_NAME" \
  "$@"
