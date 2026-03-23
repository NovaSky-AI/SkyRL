set -x

# Colocated DAPO training+generation for GLM-4.7-Flash on DAPO with Megatron.
# GLM-4.7-Flash (zai-org/GLM-4.7-Flash) is a DeepSeek-V3 architecture clone
# with MLA + MoE (64 routed experts, 4 active per token, ~3B active parameters).
#
# Runs on 1 node of 8 GPUs (TP=1 EP=8 for Megatron, 2x TP=4 vLLM engines).
# GLM-4.7-Flash has 20 attention heads, so vLLM TP must divide 20 (use TP=4).
#
# Setup:
#   1. Install deps:
#        uv sync --extra megatron
#   2. GLM-4.7-Flash needs transformers>=5.0.0 (for Glm4MoeLiteConfig).
#      If not yet available via uv sync, install manually:
#        uv pip install "transformers>=5.0.0"
#   3. Prepare data:
#        bash examples/algorithms/dapo/prepare_dapo_data.sh
#   4. Run:
#        export WANDB_API_KEY=<your_key_here>  # or set LOGGER=console below
#        bash examples/train/megatron/run_dapo_glm_flash.sh

MODEL_NAME="zai-org/GLM-4.7-Flash"
DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
NUM_NODES=2
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=4
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_ENGINE_MAX_MODEL_LEN=32000

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
# use token mean loss reduction
LOSS_REDUCTION="token_mean"
# applies overlong filtering (but not soft overlong punishment)
APPLY_OVERLONG_FILTERING=true
# apply soft overlong punishment with custom trainer impl in main_dapo.py
OVERLONG_BUFFER_LEN=$((1024 * 3))
OVERLONG_BUFFER_PENALTY_FACTOR=0

# other DAPO parameters
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 20))

# repro run parameters
TRAIN_BATCH_SIZE=128
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=32
ENFORCE_EAGER=true # cuda graphs can cause some instability
LR=1e-6

# GLM-4.7-Flash supports flash attention (v_head_dim == qk_head_dim + qk_rope_head_dim == 256).
FLASH_ATTN=true

# Megatron parallelism: TP=1, EP=8 fits 64 MoE experts across 8 GPUs (8 experts/GPU)
MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1
# MEGATRON_LAST_PIPELINE_STAGE_LAYER=23

# MoE routing flags (DeepSeek-V3 style: sigmoid scoring with expert bias)
MOE_TOKEN_DISPATCHER="alltoall"
MOE_ROUTER_LB="none"
MOE_GROUPED_GEMM=true
MOE_ROUTER_SCORE_FN="sigmoid"
MOE_ROUTER_EXPERT_BIAS=true

# CPU optimizer offload to fit in 80GB GPUs
OPTIMIZER_CPU_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# TIS parameters
TIS_IMP_RATIO_CAP=2.0
USE_TIS=true

# EFA
SKYRL_LD_LIBRARY_PATH_EXPORT=1
LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
FI_PROVIDER=efa

SKYRL_RAY_PG_TIMEOUT_IN_S=450 uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
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
  generator.inference_engine.engine_init_kwargs.max_model_len=$INFERENCE_ENGINE_MAX_MODEL_LEN \
  trainer.algorithm.use_tis=$USE_TIS \
  trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
  trainer.epochs=20 \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=40 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=false \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="dapo_glm_flash" \
  trainer.run_name="dapo_glm4_7_flash_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}" \
  trainer.export_path="$HOME/exports/dapo_glm4_7_flash_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="s3://skyrl-anyscale/org_vz1ufrqstecz2uet1xkwdzrm9b/cld_cntqf5nf645kv8esukgcy9yveg/artifact_storage/ckpts/dapo_glm4_7_flash_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}" \
  $@