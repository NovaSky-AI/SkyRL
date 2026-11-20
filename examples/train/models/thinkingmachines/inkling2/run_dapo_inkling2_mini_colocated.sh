set -x

# DAPO on DAPO-Math-17k (eval: AIME 2024 + 2025) for Inkling 2 Mini (or any Inkling2 checkpoint small enough for one
# node of 8xB200), colocated: Megatron (EP8) and NUM_INFERENCE_ENGINES vLLM engines share the 8 GPUs, taking turns.
# Same budget, data and reward shaping as the 4-node recipes (run_dapo_inkling2_{fullft,lora}_colocated.sh).
#
#   DATA_DIR=$HOME/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   MODEL_PATH=/path/to/models/Inkling-2-Mini PRESET=bf16_fft \
#     bash examples/train/models/thinkingmachines/inkling2/run_dapo_inkling2_mini_colocated.sh
#
# PRESET picks one of the configurations the 4-node recipes were run with (each knob can still be overridden):
# - bf16_fft: full fine-tuning, dual-clip PPO loss, BF16 rollout experts, no replays.
# - mxfp8_sc: full fine-tuning, MXFP8 rollout experts, R3 routing replay, sample-support replay with top-p 0.95 /
#   top-k 64, score centering, REINFORCE loss (run_dapo_inkling2_fullft_colocated.sh's defaults).
# - lora: LoRA rank 128 / alpha 32 on every linear at LR 2e-5, dual-clip PPO loss, synced as an adapter
#   (MERGE_LORA=false; vLLM serves it with its LoRA support) (run_dapo_inkling2_lora_colocated.sh's defaults).

MODEL_PATH="${MODEL_PATH:-thinkingmachines/Inkling-2-Mini}"
PRESET="${PRESET:-bf16_fft}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k-cleaned.parquet}"
TEST_FILE="${TEST_FILE:-['$DATA_DIR/aime-2024-cleaned.parquet','$DATA_DIR/aime-2025-cleaned.parquet']}"

case "$PRESET" in
  bf16_fft)
    MODE=fullft; DEFAULT_LOSS=dual_clip; DEFAULT_EXPERTS=bf16; DEFAULT_REPLAY=false; DEFAULT_LR=1e-6 ;;
  mxfp8_sc)
    MODE=fullft; DEFAULT_LOSS=reinforce; DEFAULT_EXPERTS=mxfp8; DEFAULT_REPLAY=true; DEFAULT_LR=1e-6 ;;
  lora)
    MODE=lora; DEFAULT_LOSS=dual_clip; DEFAULT_EXPERTS=bf16; DEFAULT_REPLAY=false; DEFAULT_LR=2e-5 ;;
  *) echo "unknown PRESET=$PRESET" >&2; exit 1 ;;
esac
export SKYRL_INKLING2_ROLLOUT_EXPERTS="${ROLLOUT_EXPERTS:-${SKYRL_INKLING2_ROLLOUT_EXPERTS:-$DEFAULT_EXPERTS}}"
POLICY_LOSS_TYPE="${POLICY_LOSS_TYPE:-$DEFAULT_LOSS}"
ROUTER_REPLAY="${ROUTER_REPLAY:-$DEFAULT_REPLAY}"
SAMPLE_SUPPORT_REPLAY="${SAMPLE_SUPPORT_REPLAY:-$DEFAULT_REPLAY}"
SCORE_CENTERING="${SCORE_CENTERING:-$SAMPLE_SUPPORT_REPLAY}"
if [ "$SAMPLE_SUPPORT_REPLAY" = true ]; then
  TOP_P="${TOP_P:-0.95}"
  TOP_K="${TOP_K:-64}"  # must exceed the top-p nucleus
else
  TOP_P="${TOP_P:-1.0}"
  TOP_K="${TOP_K:--1}"
fi
LR="${LR:-$DEFAULT_LR}"

NUM_GPUS_PER_NODE=8
# One engine per GPU by default (the model fits one GPU in BF16); TP2 engines halve the weights per GPU.
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE="${INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE:-1}"
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-$((NUM_GPUS_PER_NODE / INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE))}"
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION="${INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION:-0.8}"
INFERENCE_ENGINE_MAX_NUM_SEQS="${INFERENCE_ENGINE_MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
LOGGER="${LOGGER:-wandb}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-50}"

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
INFERENCE_ENGINE_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 512))
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-$((MAX_RESPONSE_LENGTH / 2))}"
REASONING_EFFORT="${REASONING_EFFORT:-0.1}"
# 128 prompts x 8 samples per step, 4 optimizer steps of 32 prompts.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-4}"
MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-$INFERENCE_ENGINE_MAX_MODEL_LEN}"

MEGATRON_TP="${MEGATRON_TP:-1}"
MEGATRON_EP="${MEGATRON_EP:-8}"

ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "max_num_batched_tokens": '"$MAX_NUM_BATCHED_TOKENS"'}'
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

MODE_ARGS=()
if [ "$MODE" = lora ]; then
  LORA_RANK="${LORA_RANK:-128}"
  LORA_ALPHA="${LORA_ALPHA:-32}"
  MERGE_LORA="${MERGE_LORA:-false}"
  MODE_ARGS=(
    trainer.policy.model.lora.rank=$LORA_RANK
    trainer.policy.model.lora.alpha=$LORA_ALPHA
    trainer.policy.model.lora.target_modules=all-linear
    trainer.policy.model.lora.share_expert_adapters="${SHARE_EXPERT_ADAPTERS:-false}"
    trainer.policy.megatron_config.lora_config.merge_lora=$MERGE_LORA
    trainer.policy.megatron_config.lora_config.normalize_moe_lora=true
    generator.inference_engine.scale_moe_lora_rank=true
    trainer.policy.model.lora.sync_mode="${LORA_SYNC_MODE:-memory}"
    trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=true
  )
else
  MODE_ARGS=(
    trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=false
  )
fi

RUN_NAME="${RUN_NAME:-inkling2_mini_${PRESET}_lr${LR}}"

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="$TEST_FILE" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.algorithm.policy_loss_type=$POLICY_LOSS_TYPE \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.clip_ratio_c=10.0 \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=1.0 \
  generator.apply_overlong_filtering=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
  trainer.policy.megatron_config.context_parallel_size=1 \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
  trainer.policy.megatron_config.moe_token_dispatcher_type="alltoall" \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="full" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method="uniform" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.remove_microbatch_padding=true \
  trainer.use_expandable_segments=true \
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
  trainer.eval_before_train="${EVAL_BEFORE_TRAIN:-true}" \
  trainer.eval_interval="${EVAL_INTERVAL:-5}" \
  trainer.ckpt_interval="${CKPT_INTERVAL:--1}" \
  trainer.resume_mode="${RESUME_MODE:-null}" \
  trainer.ckpt_path="${CKPT_PATH:-$HOME/ckpts/$RUN_NAME}" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.max_num_seqs=$INFERENCE_ENGINE_MAX_NUM_SEQS \
  generator.inference_engine.gpu_memory_utilization=$INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION \
  generator.inference_engine.enforce_eager="${ENFORCE_EAGER:-false}" \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.router_init_kwargs="$ROUTER_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.top_k=$TOP_K \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.top_k=-1 \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.megatron_config.moe_enable_routing_replay=$ROUTER_REPLAY \
  generator.inference_engine.enable_return_routed_experts=$ROUTER_REPLAY \
  trainer.algorithm.enable_sample_support_replay=$SAMPLE_SUPPORT_REPLAY \
  trainer.algorithm.enable_score_centering=$SCORE_CENTERING \
  generator.inference_engine.enable_return_sample_support_set=$SAMPLE_SUPPORT_REPLAY \
  generator.batched=false \
  generator.chat_template_kwargs.reasoning_effort=$REASONING_EFFORT \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="${PROJECT_NAME:-inkling2_dapo}" \
  trainer.run_name="$RUN_NAME" \
  "${MODE_ARGS[@]}" \
  "$@"
