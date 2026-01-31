set -x

# Colocated GRPO training+generation for Qwen3-235B-A22B on DAPO with Megatron.
# Runs on 4 nodes of 8xH100s

# bash examples/algorithms/dapo/prepare_dapo_data.sh
# bash examples/megatron/run_megatron_dapo_qwen3_235b_a22b_lora.sh

LOGGER="wandb"  # change to "console" to print to stdout

# Make sure these paths are accessible by or present on all nodes
DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
# download Qwen/Qwen3-235B-A22B-Instruct-2507 from huggingface
# `pip install huggingface_hub hf_transfer`
# `HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir ~/qwen235b`
MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"

NUM_NODES=4
NUM_GPUS=8

### Megatron configuration
# the max TP that can be used is 4, since Qwen3-235B-A22B uses Grouped Query Attention with 4 groups
MEGATRON_TP=4
MEGATRON_PP=4
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1
# Qwen3-235B-A22B has 94 blocks, so we set the last pipeline stage layer to use 16 blocks
MEGATRON_LAST_PIPELINE_STAGE_LAYER=16
FLASH_ATTN=true
# configure optimizer offloading
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

### Inference engine configuration
INFERENCE_BACKEND="vllm" # currently only vllm is supported for megatron
NUM_INFERENCE_ENGINES=2
# this is not ideal at the moment - enable inference engine pp in order to avoid this
# https://github.com/NovaSky-AI/SkyRL/issues/353
INFERENCE_ENGINE_TP=16
# the default max model len for Qwen3-235B-A22B-Instruct-2507 is 262K, and VLLM checks that 
# the KV cache memory allocated is enough to serve 1 request with max model len. Lowering to the actual
# max model len value for this script.
INFERENCE_ENGINE_MAX_MODEL_LEN=12000

# LoRA configuration
LORA_RANK=128
LORA_ALPHA=128

# DAPO parameters
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
# use token mean loss reduction
LOSS_REDUCTION="token_mean"
# applies overlong filtering (but not soft overlong punishment)
APPLY_OVERLONG_FILTERING=true
# apply soft overlong punishment with custom trainer impl in main_dapo.py
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# other DAPO parameters
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))

# repro run parameters
TRAIN_BATCH_SIZE=128
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=16
EVAL_N_SAMPLES_PER_PROMPT=32
ENFORCE_EAGER=false # cuda graphs can cause some instability
LR=1e-5 # 10x compared to full finetuning

# rollout correction parameters
TIS_RATIO_TYPE="token"
TIS_IMP_RATIO_CAP=2.0

uv run --isolated --extra mcore -m skyrl_train.entrypoints.main_base \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  +trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
  +trainer.algorithm.overlong_buffer.penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  generator.enforce_eager=$ENFORCE_EAGER \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_RATIO_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  +trainer.policy.megatron_config.transformer_config_kwargs.num_layers_in_last_pipeline_stage=$MEGATRON_LAST_PIPELINE_STAGE_LAYER \
  +generator.engine_init_kwargs.max_model_len=$INFERENCE_ENGINE_MAX_MODEL_LEN \
  trainer.use_sample_packing=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="dapo_aime" \
  trainer.run_name="dapo_qwen3_235b_a22b_base_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_ep${MEGATRON_EP}_lora_rank${LORA_RANK}_alpha${LORA_ALPHA}" \
  trainer.export_path="$HOME/exports/dapo_qwen3_235b_a22b_base_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_ep${MEGATRON_EP}_lora_rank${LORA_RANK}_alpha${LORA_ALPHA}" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="$HOME/ckpts/dapo_qwen3_235b_a22b_base_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_ep${MEGATRON_EP}_lora_rank${LORA_RANK}_alpha${LORA_ALPHA}" \
  $@