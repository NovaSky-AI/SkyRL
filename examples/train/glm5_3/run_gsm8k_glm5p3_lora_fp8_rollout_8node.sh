set -x

# Colocated GRPO on GSM8K for GLM-5.3 (753B glm_moe_dsa): bf16 Megatron LoRA trainer, online-FP8
# vLLM rollouts. 8 nodes x 8xB200, all colocated. A smoke test of the FP8 inference path
# (non-zero reward, sane rollout/train logprob gap), not a learning run.
#
#   uv run --isolated examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   export WANDB_API_KEY=<key>
#   bash examples/train/glm5_3/run_gsm8k_glm5p3_lora_fp8_rollout_8node.sh
#
# Settings shared with examples/train/glm5_3_flash/run_gsm8k_glm5p3_flash_lora_1node.sh are
# explained there. What differs:
#   * FP8 rollouts without fp8_weight_sync_mode (as in text_to_sql/run_skyrl_sql_fp8.sh): the
#     trainer syncs merged bf16 weights and vLLM's layerwise reload re-quantizes each layer as
#     its weights arrive. One TP8 engine then holds ~95 GiB/GPU of weights; bf16 would be
#     ~176 GiB and not fit a B200.
#   * merge_lora=true: vLLM serves plain merged weights, no vLLM-side LoRA.
#   * No KDA/mHC/k-pool: GLM-5.3 is plain MLA + DSA (indexer on every 4th layer) through
#     Megatron-Bridge's stock GLM5Bridge, so LoRA targets are the MLA + MLP linears only.

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/models/glm5p3-bf16}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"

NUM_NODES=8
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=8          # one TP8 FP8 engine per node
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="${LOGGER:-wandb}"

MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-5}"

# The chat template always opens <think>, so the response budget has to cover the reasoning.
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=4096
INFERENCE_ENGINE_MAX_MODEL_LEN=5120

# dp = 64/TP8 = 8; (mini_batch * n_samples) = 256 divides it.
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
MAX_TOKENS_PER_MICROBATCH=8192

USE_KL_LOSS=false
LR=1e-5
# FP8 rollouts vs the bf16 trainer widen the rollout/train logprob gap; token TIS bounds it.
TIS_TYPE="${TIS_TYPE:-token}"
TIS_CLIP_HIGH=2.0

# share_expert_adapters=true: one adapter per EP rank, shared by that rank's 256/EP local
# experts (so EP=64 gives 64 expert adapters per layer, each covering 4 experts). Rank 64 is
# ~6.2B LoRA params in total, ~0.22B per GPU. Rank 256 (~0.88B per GPU) OOMs the weight sync
# after the first step: ~11 GiB of adapter-sized fp32 state stays on the GPU after the
# optimizer offload, on top of the ~122 GiB vLLM holds during the FP8 reload.
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-$LORA_RANK}"
MERGE_LORA=true
SHARE_EXPERT_ADAPTERS=true
NORMALIZE_MOE_LORA=false
LORA_TARGET_MODULES='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2]'

# TP8 rather than the Flash recipes' TP4: it halves each GPU's share of the attention, dense and
# embedding weights (~5 GiB). At TP4 the weight sync after the first step peaked at ~173 of
# 178 GiB and OOMed: vLLM holds ~122 GiB during the FP8 reload (89 GiB weights, ~9 GiB asleep
# residual, ~24 GiB re-quantization buffers) and the trainer ~51 GiB.
MEGATRON_TP="${MEGATRON_TP:-8}"
MEGATRON_PP=1
MEGATRON_CP=1
# Frozen bf16 base is ~1.5 TB, ~1.45 TB of it routed experts. EP=64 puts 4 experts on each GPU
# (~23 GiB), keeping the trainer's share of the colocated sync as small as it can be.
MEGATRON_EP=64
MEGATRON_ETP=1
# Must stay 1 for glm_moe_dsa under EP: megatron-core splits each MoE layer's local tokens with
# torch.chunk(mlp_chunks_for_training), which yields fewer chunks on shorter ranks, so ranks issue
# different numbers of EP all-to-alls and training deadlocks (the Flash recipes' 64 is safe only
# because their mHC layer does not take that path).
MLP_CHUNKS_FOR_TRAINING=1

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0
ENABLE_ROUTING_REPLAY="${ENABLE_ROUTING_REPLAY:-false}"
# vLLM keeps its CUDA-graph pool through sleep, so it competes with the trainer during the
# colocated sync. At the default 512-sequence capture ceiling the pool was ~14 GiB and the sync
# after the first training step OOMed by ~1.5 GiB; training needs <=32 seqs/engine, eval ~165.
INFERENCE_ENGINE_MAX_NUM_SEQS=256
MAX_CUDAGRAPH_CAPTURE_SIZE=128
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.8}"

# fp8: vLLM's online per-tensor FP8 (one scale per weight, per expert for MoE). fp8_per_block
# (128x128 blocks) fails on the first weight sync with vLLM 0.30 + torch 2.13: the reload hands a
# vLLM parameter subclass to the torch.compile'd per_block_cast_to_fp8 and dynamo hits
# RecursionError in its __torch_function__.
# load_format=dummy: SkyRL syncs the trainer's weights before the first generation anyway, and a
# broken reload then shows up as garbage instead of silently keeping the checkpoint weights.
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "quantization": "'"$VLLM_QUANTIZATION"'", "load_format": "dummy", "kv_cache_dtype": "bfloat16", "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "max_cudagraph_capture_size": '"$MAX_CUDAGRAPH_CAPTURE_SIZE"', "pass_config": {"fuse_allreduce_rms": false}}}'
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'

export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=128
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export SKYRL_VLLM_START_PORT="${SKYRL_VLLM_START_PORT:-8400}"
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1

RUN_NAME="${RUN_NAME:-glm5p3_gsm8k_lora_r${LORA_RANK}_${VLLM_QUANTIZATION}_tp${MEGATRON_TP}_ep${MEGATRON_EP}}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.loss_reduction="sequence_mean" \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_CLIP_HIGH \
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
  trainer.policy.megatron_config.moe_router_score_function="sigmoid" \
  trainer.policy.megatron_config.moe_router_load_balancing_type="none" \
  trainer.policy.megatron_config.moe_enable_routing_replay=$ENABLE_ROUTING_REPLAY \
  generator.inference_engine.enable_return_routed_experts=$ENABLE_ROUTING_REPLAY \
  trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel=true \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="selective" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=[core_attn,moe] \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=null \
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=$MLP_CHUNKS_FOR_TRAINING \
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
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_interval=$MAX_TRAINING_STEPS \
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
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm5p3_gsm8k" \
  trainer.run_name="$RUN_NAME" \
  "$@"
