set -x

# Colocated GRPO training+generation for GLM-4.7-Flash on GSM8K with Megatron.
# GLM-4.7-Flash (zai-org/GLM-4.7-Flash) is a DeepSeek-V3 architecture clone
# with MLA + MoE (64 routed experts, 4 active per token, ~3B active parameters).
#
# Runs on 1 node of 8 GPUs (TP=1 EP=8 for Megatron, 2x TP=4 vLLM engines).
# GLM-4.7-Flash has 20 attention heads, so vLLM TP must divide 20 (use TP=4).
#
# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/megatron/run_megatron_grpo_glm4_7_30b.sh

MODEL_NAME="zai-org/GLM-4.7-Flash"
DATA_DIR="$HOME/data/gsm8k"
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"

NUM_NODES=1
NUM_GPUS=8

# Megatron parallelism: TP=1, EP=8 fits 64 MoE experts across 8 GPUs (8 experts/GPU)
MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

# vLLM inference: 2 engines x TP=4 = 8 GPUs (20 heads / 4 = 5 heads per GPU)
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP=4
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

# GLM-4.7-Flash supports flash attention (v_head_dim == qk_head_dim + qk_rope_head_dim == 256).
# Most other MLA models (DeepSeek-V3, Moonlight) do NOT support flash attention due to
# mismatched Q/V head dimensions. Use flash_attn=false for those models.
FLASH_ATTN=true

# MoE routing flags (DeepSeek-V3 style: sigmoid scoring with expert bias)
MOE_TOKEN_DISPATCHER="alltoall"
MOE_ROUTER_LB="none"
MOE_GROUPED_GEMM=true
MOE_ROUTER_SCORE_FN="sigmoid"
MOE_ROUTER_EXPERT_BIAS=true

# CPU optimizer offload to fit in 80GB GPUs
OPTIMIZER_CPU_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

uv run --isolated --extra mcore -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.enforce_eager=true \
  +generator.engine_init_kwargs.max_model_len=$INFERENCE_ENGINE_MAX_MODEL_LEN \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  +trainer.policy.megatron_config.transformer_config_kwargs.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER \
  +trainer.policy.megatron_config.transformer_config_kwargs.moe_router_load_balancing_type=$MOE_ROUTER_LB \
  +trainer.policy.megatron_config.transformer_config_kwargs.moe_grouped_gemm=$MOE_GROUPED_GEMM \
  +trainer.policy.megatron_config.transformer_config_kwargs.moe_router_score_function=$MOE_ROUTER_SCORE_FN \
  +trainer.policy.megatron_config.transformer_config_kwargs.moe_router_enable_expert_bias=$MOE_ROUTER_EXPERT_BIAS \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.use_sample_packing=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.algorithm.use_kl_loss=false \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.5 \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm4_7_30b_grpo" \
  trainer.run_name="glm4_7_30b_a3b_grpo_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/glm4_7_30b_a3b_grpo_megatron" \
  $@
