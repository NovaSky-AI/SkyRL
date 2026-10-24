set -x

# Colocated GRPO training+generation for DeepSeek-V4-Flash on GSM8K with Megatron, full fine-tuning.
# 2 nodes x 8xB200 (178 GiB/GPU), all colocated, against a running Ray cluster (`ray start` on both
# nodes with the same Ray version as the uv env; MODEL_PATH and DATA_DIR present on every node).
#
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   export WANDB_API_KEY=<your_key_here>   # or leave LOGGER=console
#   bash examples/train/deepseek_v4_flash/run_gsm8k_dsv4_flash_fullft_2node.sh
#
# Short-context smoke run: prompts capped at 512 tokens, responses at 1024, a 2k engine context.
#
# Memory: DeepSeek-V4-Flash is ~284B params (~530 GiB bf16, 97% of it routed experts). At TP=1 /
# EP=16 each GPU holds ~15 GiB of replicated attention/mHC weights + ~35 GiB of experts, the same
# again as bf16 grads (grad_reduce_in_fp32=false: fp32 grads would add ~50 GiB/GPU), and the fp32
# master + Adam state (~3.4 TiB) is offloaded to CPU (~1.7 TiB per node). vLLM's ~66 GiB/GPU shard
# sleeps (level 2) during training. More nodes (EP=32) leave room for fp32 grad reduction.
#
# MODEL_PATH is a BF16 copy, not the release: deepseek-ai/DeepSeek-V4-Flash ships FP8 attention and
# MXFP4 experts. Megatron-Bridge dequantizes those on import, but vLLM must serve the same BF16
# weights the trainer syncs into it. Build the copy with
#   DSV4_LOCAL_DIR=<release dir> python examples/train/deepseek_v4_flash/dev/make_slice.py <out> $(seq -s, 0 42)
# (every layer; MTP dropped; chat template embedded -- the release ships none).

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/avigyabb/models/DeepSeek-V4-Flash-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
NUM_NODES=2
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=2          # one engine per node, colocated with that node's policy shard
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="${LOGGER:-console}"  # change to "wandb" to log to wandb

# Stage 1 smoke run: an epoch is ~233 steps, and a step is ~16 min (training is kernel-launch-bound:
# small padded microbatches through mHC's Sinkhorn iterations, compressor and indexer per layer).
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-10}"

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

# 32 prompts x 8 samples = 256 sequences per step, one optimizer step per batch. DP is 16 (TP=1).
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
# Unpacked microbatches (see remove_microbatch_padding below): each holds this many padded rows.
# 2 OOM'd in the forward at step 1 (168 GiB in use: bf16 weights + grads + no-recompute activations).
MICRO_TRAIN_BATCH_SIZE_PER_GPU=1
MICRO_FORWARD_BATCH_SIZE_PER_GPU=4

USE_KL_LOSS=false   # no ref model is constructed when this (and use_kl_in_reward) are false
LR=1e-6

# Megatron mesh. megatron-core's DSv4 hybrid attention is TP=1-only at the pinned rev, so attention
# is replicated and EP shards the experts (16/GPU). PP stays 1 (hash-routed layers must sit on the
# embedding's stage; mHC + PP is untested here).
MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=16
MEGATRON_ETP=1

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Rollout router replay (R3). Off for this smoke run; both knobs move together.
ENABLE_ROUTING_REPLAY=false

# vLLM's share of each GPU: 0.6 -> ~107 GiB, ~66 GiB of it the BF16 shard at TP8. Megatron's weights
# stay resident (~50 GiB) through rollout; grads and activations only exist while vLLM sleeps.
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.6}"

ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "enable_prefix_caching": false}'

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

RUN_NAME="dsv4_flash_gsm8k_fullft_tp${MEGATRON_TP}_ep${MEGATRON_EP}"

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
  `# Selective recompute of the MoE layers: megatron-core's DSv4 attention rejects core_attn` \
  `# checkpointing, and full recompute is incompatible with mHC, so MoE is what remains.` \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="selective" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[moe] \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null \
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=64 \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.disable_parameter_transpose_cache=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
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
