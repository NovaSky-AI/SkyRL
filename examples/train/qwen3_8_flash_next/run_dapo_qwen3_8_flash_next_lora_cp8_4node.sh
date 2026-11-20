set -x

# Colocated sync DAPO for Qwen3.8-Flash-Next (HF qwen4_exp, ~180B MoE) with Megatron + LoRA and
# context parallelism (TP1 x CP8 x EP16, DP4), with 64K-token responses. Four nodes of 8xB200 (32 GPUs).
# Stops after MAX_TRAINING_STEPS.
#
#   bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   export WANDB_API_KEY=<key>
#   MODEL_PATH=/path/to/Qwen3.8-Flash-Next bash examples/train/qwen3_8_flash_next/run_dapo_qwen3_8_flash_next_lora_cp8_4node.sh
#
# Per-setting reasoning (parallelism, vLLM memory, NVLS, frozen modules) lives
# in run_gsm8k_qwen3_8_flash_next_lora_4node.sh. What differs here: the stock DAPO 2k prompt / 8k
# response budget, which runs the QSA indexer in its sparse regime (sequences past
# indexer_budget + compress_ratio - 1 = 2051 tokens), and rollout router replay (R3).
#
# Context parallelism (CP8) splits each sequence over 8 GPUs, so responses can run far past the 8K of
# run_dapo_qwen3_8_flash_next_lora_sync_4node.sh: 64K here, and up to 262,144 tokens of prompt +
# response (raise MAX_RESPONSE_LENGTH; the vLLM max_model_len and micro-batch follow it).
# CP for qwen4_exp covers the QSA attention (split-K cuDNN QSA over CP chunks with an LSE merge), the
# PLE halo and the QSA indexer's pooled-key all-gather; GDN CP is megatron-core's. CP needs Blackwell
# and the cuDNN QSA kernels (QSA_CUDNN_WHEEL below), and TP x CP <= 16 (the GDN key-head count).
# If the image puts its own cuDNN on LD_LIBRARY_PATH (e.g. /opt/cudnn), the Ray workers load those
# sublibraries instead of the pip cuDNN and QSA runs on that older version: drop that entry and set
# SKYRL_LD_LIBRARY_PATH_EXPORT=1 so the workers get the driver's path.
# Measured on 4 x 8 B200 with the defaults (2-step smoke): 562-1,015 s per step, most of it generation
# once responses reach the 64K cap; weight sync 11 s; peak 36.8 GB allocated; rollout/trainer |dlogp|
# 0.013-0.014. The base model already scores ~0.95 on AIME-2024 here, so DAPO-math gives little signal.

# On B200, the sparse-regime attention runs on cuDNN frontend's QSA kernels (1.7-3.5x FlexAttention)
# when a cudnn-frontend with cudnn.ops.qsa_forward is installed: >= 1.32, until then the
# nvidia_cudnn_frontend-1.32.0.dev* cp312 wheel from github.com/NVIDIA/cudnn-frontend/releases.
# Set QSA_CUDNN_WHEEL to its path (the same path on every node; Ray workers reuse the `uv run`
# command line). CP > 1 requires it.
QSA_CUDNN_WHEEL="${QSA_CUDNN_WHEEL:-}"

export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=128
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# Otherwise redirect_actor_output_to_file() swallows vLLM's errors.
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
# fla's TileLang GDN backward aborts on Blackwell; these recipes target B200, so default to Triton.
export FLA_TILELANG="${FLA_TILELANG:-0}"

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/Qwen3.8-Flash-Next}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k-cleaned.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/aime-2024-cleaned.parquet}"

NUM_NODES=4
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="${LOGGER:-wandb}"

MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-20}"

# Trainer parallelism. TP is 1 or 2 (TP=2 runs with sequence parallelism). At TP=4 a rank has
# 6 query heads per KV head, which the cuDNN QSA kernels don't cover.
MEGATRON_TP="${MEGATRON_TP:-1}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-8}"
MEGATRON_EP="${MEGATRON_EP:-16}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"

# Chunks per MoE layer call during training. Each chunk repeats the routing, the expert all-to-all
# and the kernel launches. Measured on 2 x 8 B200 (CP8, EP16), 1 chunk against 8: at 66K-token
# sequences policy_train 12.2 s -> 6.5 s for +3.6 GB peak; at 250K the time is about equal and the
# peak is +16 GB (89 GB allocated). Raise it only to save memory.
MLP_CHUNKS_FOR_TRAINING="${MLP_CHUNKS_FOR_TRAINING:-1}"

if [ "$MEGATRON_CP" -gt 1 ] && [ -z "$QSA_CUDNN_WHEEL" ]; then
  echo "QSA_CUDNN_WHEEL must point at the cudnn-frontend wheel when MEGATRON_CP > 1 (split-K cuDNN QSA)." >&2
  exit 1
fi

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-65536}"
# prompt + response + chat-template headroom; at most 262,144.
INFERENCE_ENGINE_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 512))
OVERLONG_BUFFER_LEN=4096                     # penalty starts 4,096 tokens before the cap
# Half-strength soft overlong penalty. At 1.0, full FT learned to cut its reasoning short: by step 20,
# held-out AIME accuracy on finished responses fell to 0.86 (0.99 at 0.5) with 33/240 finished-wrong.
OVERLONG_BUFFER_PENALTY_FACTOR="${OVERLONG_BUFFER_PENALTY_FACTOR:-0.5}"

# dp = 32 / (TP * PP * CP), and (mini_batch * n_samples) % dp == 0.
# 16 prompts x 8 samples = 128 sequences per step, one optimizer step.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=12
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"   # held-out AIME-2024 at steps 0, 5, 10, ...
MAX_TOKENS_PER_MICROBATCH=$INFERENCE_ENGINE_MAX_MODEL_LEN  # must hold one full sequence

# Rollout router replay (R3): vLLM returns the experts each token was routed to and Megatron
# replays them, so the trainer scores rollouts with the sampler's routing (512 experts, top-10).
ENABLE_ROUTING_REPLAY="${ENABLE_ROUTING_REPLAY:-true}"

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
# LoRA needs a much larger LR than full FT: at 1e-5 / 3e-5 the per-step update was ~1e-4 nats/token
# and runs stayed flat; 3e-4 with alpha 32 learns (held-out AIME 0.45 -> 0.85 in 10 steps).
LR="${LR:-3e-4}"

# Rollout weights. With MERGE_LORA=true the trainer sends merged weights to vLLM on every sync, and
# vLLM decodes without the adapter kernels. The merged sync takes ~28 s against ~9 s adapter-only,
# and decode is ~1.9x faster, so merging pays when the longest response is past ~3.6K tokens.
# Measured on 2 x 8 B200 with this 2k/64K budget (4 prompts x 8): RL step 866 s -> 510 s with a
# response at the 64K cap, 290 s -> 205 s with a 24-29K longest response. With long prompts and
# short responses merging loses (66K prompts, 1K responses: 33 s -> 47 s per step): set
# MERGE_LORA=false LORA_SYNC_MODE=memory there. Merged sync needs lora.sync_mode=disk.
MERGE_LORA="${MERGE_LORA:-true}"
LORA_SYNC_MODE="${LORA_SYNC_MODE:-disk}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
# vLLM's custom all-reduce registers CUDA-graph buffers over CUDA IPC, which fails on sleep mode's cumem
# allocator (colocate_all=true, TP>1): every TP worker dies at engine start with
# custom_all_reduce.cuh:164 'invalid argument'.
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "disable_custom_all_reduce": true}'
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'

RUN_NAME="${RUN_NAME:-qwen3_8_flash_next_dapo_sync_lora_r16_64k_cp8_ep16}"
CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/$RUN_NAME}"

uv run --isolated --extra megatron ${QSA_CUDNN_WHEEL:+--with "$QSA_CUDNN_WHEEL"} -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.mtp_num_layers=0 \
  trainer.policy.megatron_config.moe_token_dispatcher_type="alltoall" \
  trainer.policy.megatron_config.moe_enable_routing_replay=$ENABLE_ROUTING_REPLAY \
  generator.inference_engine.enable_return_routed_experts=$ENABLE_ROUTING_REPLAY \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="full" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method="uniform" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=$MLP_CHUNKS_FOR_TRAINING \
  trainer.policy.model.lora.rank=16 \
  trainer.policy.model.lora.alpha=32 \
  trainer.policy.model.lora.sync_mode=$LORA_SYNC_MODE \
  trainer.policy.megatron_config.lora_config.merge_lora=$MERGE_LORA \
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
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=true \
  trainer.eval_interval=$EVAL_INTERVAL \
  trainer.ckpt_interval=10 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.distributed_executor_backend="mp" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.max_num_seqs=128 \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.router_init_kwargs="$ROUTER_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  environment.env_class=aime \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen3_8_flash_next_dapo" \
  trainer.run_name="$RUN_NAME" \
  "$@"
