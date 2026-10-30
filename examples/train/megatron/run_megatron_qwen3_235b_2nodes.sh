set -x

# Colocated DAPO+LoRA training+generation for Qwen3-235B-A22B on 2 nodes of 8xH100s.
# Adapted from run_megatron_dapo_qwen3_235b_a22b_lora.sh (4-node) for 16-GPU setups.
# Uses FP8 training (TransformerEngine) and FP8 vLLM inference to reduce VRAM pressure.

# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/megatron/run_megatron_qwen3_235b_2nodes.sh

LOGGER="console"  # set to "wandb" with WANDB_API_KEY env var to enable wandb

# Use shared cluster storage so workers can read the data files too.
DATA_DIR="/mnt/cluster_storage/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k.parquet"
TEST_FILE="$DATA_DIR/aime-2024.parquet"

mkdir -p "$DATA_DIR"
[ -f "$TRAIN_FILE" ] || wget -q -O "$TRAIN_FILE" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
[ -f "$TEST_FILE" ] || wget -q -O "$TEST_FILE" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
# download Qwen/Qwen3-235B-A22B-Instruct-2507 from huggingface
# `pip install huggingface_hub hf_transfer`
# `HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir ~/qwen235b`
MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"

NUM_NODES=2
NUM_GPUS_PER_NODE=8

### Megatron configuration
# Qwen3-235B-A22B uses GQA with 4 groups, so max TP=4
MEGATRON_TP=4
# PP=2 (halved from 4-node config) keeps DP=2: TP x PP = 8, 16/8 = 2 DP
MEGATRON_PP=2
MEGATRON_CP=1
# EP=2 (EP must divide DP; DP=2 so EP<=2)
MEGATRON_EP=2
MEGATRON_ETP=1
# Qwen3-235B-A22B has 94 blocks; with PP=2, each stage holds 47 blocks
MEGATRON_LAST_PIPELINE_STAGE_LAYER=47
FLASH_ATTN=true
# Optimizer offloading is essential for fitting 235B on 16 GPUs
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

### Inference engine configuration
INFERENCE_BACKEND="vllm"
# 2 engines × TP=8 (each engine on a single node) avoids cross-node NCCL
# in vLLM init, which was hanging silently.
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP=8
# Lowered from 12000 to reduce KV cache VRAM on 16 GPUs
INFERENCE_ENGINE_MAX_MODEL_LEN=8192

### LoRA configuration
LORA_RANK=128
LORA_ALPHA=128

### DAPO parameters
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))

### Batch sizes (conservative for 16-GPU memory budget)
TRAIN_BATCH_SIZE=64      # halved from 128
MINI_BATCH_SIZE=16       # halved from 32
N_SAMPLES_PER_PROMPT=16
EVAL_N_SAMPLES_PER_PROMPT=32
ENFORCE_EAGER=true  # skip CUDA-graph capture — vLLM 235B+FP8 was hanging silently during capture
LR=1e-5

### Rollout correction
TIS_RATIO_TYPE="token"
TIS_IMP_RATIO_CAP=2.0

# Disable Ray's uv-run propagation hook so worker actors use the shared venv
# directly via py_executable instead of re-running uv install per actor.
unset RAY_RUNTIME_ENV_HOOK

# Forward SIGTERM/SIGINT to the python child so its in-process signal handler can
# call dist.destroy_process_group() + cuda.synchronize() before k8s sends SIGKILL.
trap 'echo "[script] forwarding SIGTERM to pid=$PID"; kill -TERM "$PID" 2>/dev/null; wait "$PID"' TERM INT

# Ray workers share the head's .venv (built from this working_dir), so we must
# keep --extra megatron. MAX_JOBS=1 / NINJA_JOBS=1 in env_vars caps build memory.
/mnt/cluster_storage/.skyrl-venv/bin/python -m examples.train.algorithms.dapo.main_dapo \
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
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
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
  trainer.policy.megatron_config.transformer_config_kwargs.num_layers_in_last_pipeline_stage=$MEGATRON_LAST_PIPELINE_STAGE_LAYER \
  trainer.policy.megatron_config.transformer_config_kwargs.fp8=e4m3 \
  trainer.policy.megatron_config.transformer_config_kwargs.fp8_margin=0 \
  trainer.policy.megatron_config.transformer_config_kwargs.fp8_amax_history_len=1024 \
  trainer.policy.megatron_config.transformer_config_kwargs.fp8_amax_compute_algo=max \
  generator.inference_engine.engine_init_kwargs.max_model_len=$INFERENCE_ENGINE_MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.quantization=fp8 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.use_sample_packing=true \
  trainer.flash_attn=$FLASH_ATTN \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.epochs=20 \
  trainer.eval_batch_size=512 \
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
  trainer.logger="$LOGGER" \
  trainer.project_name="dapo_aime" \
  trainer.run_name="dapo_qwen3_235b_a22b_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_ep${MEGATRON_EP}_lora_rank${LORA_RANK}_fp8_2nodes" \
  trainer.export_path="$HOME/exports/dapo_qwen3_235b_a22b_fp8_2nodes" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="$HOME/ckpts/dapo_qwen3_235b_a22b_fp8_2nodes" \
  "$@" &
PID=$!
wait "$PID"
