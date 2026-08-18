set -x

# GRPO (DAPO-style) LoRA training for moonshotai/Kimi-K2.7-Code -- a 1T-param
# DeepSeek-V3-architecture MoE (61 layers, 384 experts, MLA) released as a
# KimiK25ForConditionalGeneration checkpoint with INT4 QAT experts
# (compressed-tensors pack-quantized, group_size=32, Kimi convention
# scale_divisor=7.0 / q_min=-7) and 262144-token context.
#
# TWO 8-GPU nodes (e.g. 8xB300 each), colocated. vLLM serves the release INT4
# checkpoint (one engine per node, TP=8) with the LoRA adapter hot-loaded from
# disk each step (merge_lora=false); the Megatron trainer holds BF16 masters
# dequantized from the INT4 release and fake-quantizes the frozen expert GEMMs
# onto the same INT4 grid in the forward pass (STE backward). For Kimi QAT
# checkpoints the dequantized masters are a fixed point of the fake-quant, so
# trainer experts are bit-exact with what vLLM serves; TIS corrects the
# residual off-policy mismatch. The model is trained text-only
# (language_model_only): the vision tower stays frozen in vLLM.
#
# One-time setup:
#
# 1) Dequantize the INT4 release to BF16 masters ON EVERY NODE (or a shared
#    filesystem). ~595 GB in, ~2.1 TB out; verifies bit-exactness:
#    uv run --isolated examples/train/megatron/dequantize_compressed_tensors_int4.py \
#        --input <path-to-Kimi-K2.7-Code-snapshot> \
#        --output /data/skyrl/models/Kimi-K2.7-Code-BF16
#
# 2) Download data:
#    bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#
# 3) Start the Ray cluster (same repo checkout + venv path on both nodes), then
#    run this script on the head node only:
#    export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
#    head:   ray start --head --port=6379
#    worker: ray start --address=<head-ip>:6379
#
# The task here is math (DAPO-17k / AIME) to reuse the standard data prep; swap
# data.train_data / environment.env_class for long-horizon code tasks to make
# use of the full 262k context.

# INT4 actor served by vLLM; BF16 masters loaded by the trainer (Megatron-Bridge
# cannot load compressed-tensors, so it reads BF16 from FAKE_QUANT_BF16_PATH).
MODEL_NAME="${MODEL_NAME:-moonshotai/Kimi-K2.7-Code}"
FAKE_QUANT_BF16_PATH="${FAKE_QUANT_BF16_PATH:-/data/skyrl/models/Kimi-K2.7-Code-BF16}"

DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"

# --- TWO 8-GPU nodes, colocated. num_policy_gpus (16) == num_engines*TP (2*8). ---
# One vLLM engine per node (TP=8 fits the INT4 checkpoint comfortably), mp
# executor within each node -- no cross-node engine traffic. The trainer spans
# both nodes: TP=2 x CP=8 (DP=1), experts sharded EP=16 (24 experts/GPU/layer).
NUM_NODES=2
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-2}"
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE="${INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE:-8}"
LOGGER="${LOGGER:-wandb}"

# --- QAT / TIS ablation toggle (see the Qwen3.6 INT4 example for the study) ---
QAT_MODE="${QAT_MODE:-on}"   # on | off
if [ "$QAT_MODE" = "on" ]; then
  FAKE_QUANT_ENABLED=true
  TIS_TYPE=token
  RUN_SUFFIX="int4qat_tis_ON"
else
  FAKE_QUANT_ENABLED=false
  TIS_TYPE=null            # disables off_policy_correction TIS
  RUN_SUFFIX="int4qat_tis_OFF"
fi

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=false
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0

# Full 262144-token context: prompt budget + response budget == max_model_len.
MAX_MODEL_LEN=$((262144))
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-$((1024 * 8))}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((MAX_MODEL_LEN - MAX_PROMPT_LENGTH))}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
EVAL_N_SAMPLES_PER_PROMPT=8
ENFORCE_EAGER=false
LR=1e-5

LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
# MLA attention: mcore's q/kv low-rank projections have no "linear_qkv" module
# and vLLM's MLA weight absorption doesn't run adapters on kv_b_proj, so LoRA
# targets attention-out + all MLPs (dense, shared expert, and the 384 routed
# experts -- the routed experts carry ~96% of the parameters).
LORA_TARGET_MODULES="['linear_proj','linear_fc1','linear_fc2']"
# Capacity-normalized expert LoRA (rank/topk for the grouped experts). Without
# this, the per-expert PEFT export at rank 32 is ~41 GB for 384 experts x 60
# layers -- gathered, written, and re-read by every engine each weight sync.
NORMALIZE_MOE_LORA="${NORMALIZE_MOE_LORA:-true}"

# megatron config (16 GPUs: TP=2 x CP=8 x DP=1, EP=16/ETP=1, PP=1)
# BF16 masters/GPU: ~127 GB experts (60 layers x 24 experts) + ~12 GB rest.
MEGATRON_TP="${MEGATRON_TP:-2}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-8}"
MEGATRON_EP="${MEGATRON_EP:-16}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"

TIS_IMP_RATIO_CAP=2.0

# LoRA-only optimizer state is small; offload stays available for tight fits.
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-false}"
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Kimi K2.5-family checkpoints train text-only on Megatron (vision tower stays
# frozen in vLLM); required for KimiK25ForConditionalGeneration.
LANGUAGE_MODEL_ONLY=True
# mla_prefill_backend=FLASHINFER: the default FLASH_ATTN (FA4 CuTe-DSL) MLA
# prefill crashes against the locked nvidia-cutlass-dsl MLIR bindings; the
# FlashInfer ragged prefill has sm103 kernels via the cu130 jit-cache.
ENGINE_INIT_KWARGS="{\"max_model_len\": $MAX_MODEL_LEN, \"compilation_config\": {\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}, \"attention_config\": {\"mla_prefill_backend\": \"FLASHINFER\"}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
# Giant-model wake-up + 262k prefills need generous execute timeouts.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"

# CUDA-13 stack (required on Blackwell-Ultra/B300, see pyproject): every CUDA
# library comes from pip, so run in the project venv (no --isolated -- the env
# below points into it and must be stable across nodes) and resolve libraries
# from the pip cu13 set. CUDNN_PATH pins TE's cuDNN search to the pip cuDNN
# (a system cuDNN core mixed with pip sublibraries fails with
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED). Export these (plus
# SKYRL_LD_LIBRARY_PATH_EXPORT=1) before `ray start` on every node.
SP="$(pwd)/.venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cusparselt/lib:$SP/nvidia/nccl/lib:$SP/nvidia/nvshmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDNN_PATH="$SP/nvidia/cudnn"
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
# Ray workers execute `uv run` from Ray's *copied* working dir (no .venv
# there); pin the project env by absolute path so --no-sync finds it.
export UV_PROJECT_ENVIRONMENT="$(pwd)/.venv"

# --no-sync: the venv is prepared once per node before `ray start`; Ray's uv
# hook re-runs this exact command per worker, and concurrent implicit syncs
# from many workers can race on the shared venv.
uv run --no-sync --extra megatron -m examples.train.algorithms.dapo.main_dapo \
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
  trainer.policy.model.fake_int4_qat.enabled=$FAKE_QUANT_ENABLED \
  trainer.policy.model.fake_int4_qat.group_size=32 \
  trainer.policy.model.fake_int4_qat.scale_divisor=7.0 \
  trainer.policy.model.fake_int4_qat.q_min=-7 \
  trainer.policy.model.fake_int4_qat.bf16_base_path="$FAKE_QUANT_BF16_PATH" \
  trainer.policy.megatron_config.lora_config.merge_lora=false \
  trainer.fused_lm_head_logprob=true \
  trainer.flash_attn=false \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  generator.inference_engine.distributed_executor_backend="mp" \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.model.lora.target_modules="$LORA_TARGET_MODULES" \
  trainer.policy.megatron_config.lora_config.normalize_moe_lora=$NORMALIZE_MOE_LORA \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.epochs=1 \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=0 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  trainer.logger="$LOGGER" \
  trainer.project_name="kimi_k2_7_dapo_lora_int4qat" \
  trainer.run_name="dapo_lora_r${LORA_RANK}_kimi_k2.7_code_2node_${RUN_SUFFIX}" \
  trainer.export_path="$HOME/exports/dapo_lora_kimi_k2_7_${RUN_SUFFIX}" \
  trainer.hf_save_interval=0 \
  trainer.resume_mode=none \
  trainer.max_ckpts_to_keep=1 \
  trainer.ckpt_path="$HOME/ckpts/dapo_lora_kimi_k2_7_${RUN_SUFFIX}" \
  $@
