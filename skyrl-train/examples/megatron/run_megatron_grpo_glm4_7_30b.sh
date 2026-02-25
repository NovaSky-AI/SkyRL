set -x

# Colocated GRPO training+generation for GLM4.7-30B-A3B on GSM8K with Megatron.
# GLM4.7-30B-A3B is a DeepSeek-V3 architecture clone with MLA + MoE
# (128 experts, 8 active per token, ~3B active parameters).
#
# Runs on 2 nodes of 8xH100s (16 GPUs total).
#
# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/megatron/run_megatron_grpo_glm4_7_30b.sh

# Replace with the GLM4.7-30B-A3B model path or HF ID
MODEL_NAME="THUDM/GLM-4-9B-0414"
DATA_DIR="$HOME/data/gsm8k"
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"

NUM_NODES=2
NUM_GPUS=8

# Megatron parallelism: TP=2, EP=8 fits 128 MoE experts across 16 GPUs
MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP=8

# MLA does not support flash attention in TransformerEngine
FLASH_ATTN=false

# MoE routing flags (DeepSeek-V3 style: sigmoid scoring with expert bias)
MOE_TOKEN_DISPATCHER="alltoall"
MOE_ROUTER_LB="none"
MOE_GROUPED_GEMM=true
MOE_ROUTER_SCORE_FN="sigmoid"
MOE_ROUTER_EXPERT_BIAS=true

uv run --isolated --extra mcore -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER \
  trainer.policy.megatron_config.moe_router_load_balancing_type=$MOE_ROUTER_LB \
  trainer.policy.megatron_config.moe_grouped_gemm=$MOE_GROUPED_GEMM \
  trainer.policy.megatron_config.moe_router_score_function=$MOE_ROUTER_SCORE_FN \
  trainer.policy.megatron_config.moe_router_enable_expert_bias=$MOE_ROUTER_EXPERT_BIAS \
  trainer.ref.megatron_config.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER \
  trainer.ref.megatron_config.moe_router_load_balancing_type=$MOE_ROUTER_LB \
  trainer.ref.megatron_config.moe_grouped_gemm=$MOE_GROUPED_GEMM \
  trainer.ref.megatron_config.moe_router_score_function=$MOE_ROUTER_SCORE_FN \
  trainer.ref.megatron_config.moe_router_enable_expert_bias=$MOE_ROUTER_EXPERT_BIAS \
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
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.6 \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm4_7_30b_grpo" \
  trainer.run_name="glm4_7_30b_a3b_grpo_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/glm4_7_30b_a3b_grpo_megatron" \
  $@
