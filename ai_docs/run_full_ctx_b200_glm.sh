set -x

# Full context length testing for GLM-4.7-Flash on 1x8xB200 (183GB each)
# Tests max-length dummy sequences to find OOM boundaries.
#
# Usage:
#   # Test with specific context length (prompt + response):
#   MAX_PROMPT_LENGTH=8192 MAX_GENERATE_LENGTH=57344 bash ai_docs/run_full_ctx_b200_glm.sh
#
#   # This tests total context = 8192 + 57344 = 65536 (64K)

MODEL_NAME="zai-org/GLM-4.7-Flash"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"

# Context length config (override via env vars)
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_GENERATE_LENGTH=${MAX_GENERATE_LENGTH:-57344}
TOTAL_CTX=$((MAX_PROMPT_LENGTH + MAX_GENERATE_LENGTH))
echo "=== Testing total context length: ${TOTAL_CTX} (prompt=${MAX_PROMPT_LENGTH} + response=${MAX_GENERATE_LENGTH}) ==="

# Megatron parallelism: TP=1, EP=8 (default, no context parallelism)
# Override MEGATRON_CP, MEGATRON_EP via env vars for CP experiments
MEGATRON_TP=${MEGATRON_TP:-1}
MEGATRON_PP=${MEGATRON_PP:-1}
MEGATRON_CP=${MEGATRON_CP:-1}
MEGATRON_EP=${MEGATRON_EP:-8}
MEGATRON_ETP=${MEGATRON_ETP:-1}

# vLLM inference: 2 engines x TP=4
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP=4
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

# MoE config (same as working GLM training)
FLASH_ATTN=true
MOE_TOKEN_DISPATCHER="alltoall"
MOE_ROUTER_LB="none"
MOE_GROUPED_GEMM=true
MOE_ROUTER_SCORE_FN="sigmoid"
MOE_ROUTER_EXPERT_BIAS=true

# CPU optimizer offload
OPTIMIZER_CPU_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Batch sizes: micro=1 for max context testing
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
POLICY_MINI_BATCH_SIZE=${POLICY_MINI_BATCH_SIZE:-8}
MICRO_FORWARD_BATCH_SIZE=1
MICRO_TRAIN_BATCH_SIZE=1
N_SAMPLES_PER_PROMPT=1

# Number of dummy steps
NUM_DUMMY_STEPS=${NUM_DUMMY_STEPS:-3}

uv run --frozen --extra megatron -m examples.train_scripts.full_context.main_full_ctx \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=8 \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.enforce_eager=true \
  generator.inference_engine.engine_init_kwargs.max_model_len=$INFERENCE_ENGINE_MAX_MODEL_LEN \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER \
  trainer.policy.megatron_config.moe_router_load_balancing_type=$MOE_ROUTER_LB \
  trainer.policy.megatron_config.moe_grouped_gemm=$MOE_GROUPED_GEMM \
  trainer.policy.megatron_config.moe_router_score_function=$MOE_ROUTER_SCORE_FN \
  trainer.policy.megatron_config.moe_router_enable_expert_bias=$MOE_ROUTER_EXPERT_BIAS \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.use_sample_packing=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=1 \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=false \
  trainer.eval_interval=999 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BATCH_SIZE \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE \
  trainer.ckpt_interval=999 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.logger="console" \
  trainer.project_name="glm4_full_ctx_b200" \
  trainer.run_name="ctx${TOTAL_CTX}_tp${MEGATRON_TP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}" \
  trainer.resume_mode=null \
  trainer.num_dummy_steps=$NUM_DUMMY_STEPS \
  $@
