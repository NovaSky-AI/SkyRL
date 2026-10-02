set -x

# Fully async DAPO for DeepSeek-R1-Distill-Qwen-1.5B on DAPO training data, validated on AIME 2024,
# on the Megatron (default) or FSDP backend. Mirrors the fully async recipe of the runs in the
# `dapo-async-megatron` wandb project: async dynamic sampling (`sample_full_batch` with the
# zero-variance filter), prompt-mean loss aggregation, no KV-cache clearing on weight sync, and no
# importance-sampling masks, so the training/inference mismatch is left uncorrected unless the
# policy loss or score centering handles it. 4 training GPUs (data parallel) + 4 inference engines.
#
# Toggles (environment variables):
#   STRATEGY=megatron|fsdp                  training backend (default: megatron)
#   POLICY_LOSS=reinforce|dppo|rollout_is   policy loss (default: reinforce, plain policy gradient)
#   FP8_ROLLOUTS=true                       serve rollouts with vLLM online dynamic FP8 quantization
#   SCORE_CENTERING=true                    subtract the sampler-expected score (arXiv 2609.20807)
#   SCORE_CENTERING_TOP_K=32                sampler head size recorded per generated token
#
# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/algorithms/dapo/run_dapo_deepseek_1.5b_aime_fully_async.sh
# FP8_ROLLOUTS=true SCORE_CENTERING=true bash examples/train/algorithms/dapo/run_dapo_deepseek_1.5b_aime_fully_async.sh

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
NUM_NODES=1
NUM_GPUS_PER_NODE=4
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1
LOGGER="wandb"  # change to "console" to print to stdout
: "${PROJECT_NAME:=dapo-async}"

: "${STRATEGY:=megatron}"
: "${POLICY_LOSS:=reinforce}"
: "${FP8_ROLLOUTS:=false}"
: "${SCORE_CENTERING:=false}"
: "${SCORE_CENTERING_TOP_K:=32}"
#   NUM_WARMUP_STEPS=160                    LR warmup steps (0 disables warmup)
#   RUN_SUFFIX=                             appended to the run name (e.g. -noWarmup)
: "${NUM_WARMUP_STEPS:=160}"
: "${RUN_SUFFIX:=}"

# Backend-specific arguments. Both backends run pure data parallelism over the training GPUs.
if [ "$STRATEGY" = "megatron" ]; then
  MEGATRON_TP=1
  MEGATRON_PP=1
  MEGATRON_CP=1
  BACKEND_ARGS=(
    trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP
    trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP
    trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP
  )
else
  BACKEND_ARGS=(trainer.policy.fsdp_config.fsdp_size=$NUM_GPUS_PER_NODE)
fi

# DAPO settings shared with the Qwen3 fully async example.
LOSS_REDUCTION="prompt_mean"
APPLY_OVERLONG_FILTERING=true
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0
USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
# Only read by the clipping-based losses (rollout_is); reinforce and dppo ignore them.
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))
N_SAMPLES_PER_PROMPT=16
EVAL_N_SAMPLES_PER_PROMPT=32
ENFORCE_EAGER=true
LR=1e-6

# Fully async knobs.
: "${MAX_STALENESS_STEPS:=8}"
: "${NUM_PARALLEL_GENERATION_WORKERS:=144}"
: "${MINI_BATCH_SIZE:=32}"
: "${EVAL_CKPT_INTERVAL:=80}"

FP8_ARGS=()
FP8_TAG=""
if [ "$FP8_ROLLOUTS" = "true" ]; then
  FP8_ARGS=(generator.inference_engine.engine_init_kwargs.quantization=fp8)
  FP8_TAG="-fp8"
fi
SC_TAG=""
if [ "$SCORE_CENTERING" = "true" ]; then
  SC_TAG="-scoreCenter${SCORE_CENTERING_TOP_K}"
fi

RUN_NAME=dapo_deepseek_r1_distill_1.5b-async-${STRATEGY}-${POLICY_LOSS}${FP8_TAG}${SC_TAG}-promptMean-dynSample-noMask-bs${MINI_BATCH_SIZE}-maxStale${MAX_STALENESS_STEPS}-numCon${NUM_PARALLEL_GENERATION_WORKERS}-${NUM_GPUS_PER_NODE}train${NUM_INFERENCE_ENGINES}gen${RUN_SUFFIX}

uv run --isolated --extra $STRATEGY -m examples.train.algorithms.dapo.main_dapo_fully_async \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.fully_async.enabled=true \
  trainer.fully_async.max_staleness_steps=${MAX_STALENESS_STEPS} \
  trainer.fully_async.num_parallel_generation_workers=${NUM_PARALLEL_GENERATION_WORKERS} \
  trainer.fully_async.clear_kv_cache_on_weight_sync=false \
  trainer.fully_async.sample_full_batch=true \
  trainer.algorithm.zero_variance_filter=true \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="$POLICY_LOSS" \
  trainer.algorithm.score_centering.enabled=$SCORE_CENTERING \
  trainer.algorithm.score_centering.top_k=$SCORE_CENTERING_TOP_K \
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
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=false \
  trainer.strategy=$STRATEGY \
  trainer.remove_microbatch_padding=true \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  "${BACKEND_ARGS[@]}" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=$EVAL_CKPT_INTERVAL \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=$EVAL_CKPT_INTERVAL \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=$NUM_WARMUP_STEPS \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.export_path="$HOME/exports/$RUN_NAME" \
  trainer.hf_save_interval=0 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="$HOME/ckpts/$RUN_NAME" \
  "${FP8_ARGS[@]}" \
  "$@"
