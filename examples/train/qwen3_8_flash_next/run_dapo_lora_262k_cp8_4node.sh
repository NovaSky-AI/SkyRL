set -x

# Colocated sync DAPO for Qwen3.8-Flash-Next (HF qwen4_exp, ~180B MoE) with Megatron + LoRA at
# 262,144-token sequences, with context parallelism (TP1 x CP8 x EP16, DP4). Four nodes of 8xB200.
# Stops after MAX_TRAINING_STEPS.
#
#   Data: KeyChain long-context multi-hop QA (keychain/make_keychain.py, after LoongRL), ~65.5K-token
#   prompts, filtered to the ones the base model sometimes but not always answers (~20 min on 4 nodes):
#   M=/path/to/Qwen3.8-Flash-Next; D=$HOME/data/keychain; K=examples/train/qwen3_8_flash_next/keychain
#   uv run --isolated --extra megatron python $K/make_keychain.py --model-path $M --out-dir $D
#   for s in train val; do  # one node; on N nodes run shard i of N per node (--shard i --num-shards N)
#     uv run --isolated --extra megatron python $K/calibrate_keychain.py --model-path $M --pool $D/pool_$s.parquet --out $D/calib_$s.jsonl
#     uv run --isolated --extra megatron python $K/filter_keychain.py $D/pool_$s.parquet $D/keychain_64k_${s}_filtered.parquet $D/calib_$s.jsonl
#   done
#   export WANDB_API_KEY=<key>
#   MODEL_PATH=$M bash examples/train/qwen3_8_flash_next/run_dapo_lora_262k_cp8_4node.sh
#
# Rows set env_class=search and max_turns=1: one answer per prompt, scored by exact match, no retrieval.
# Measured on 4 x 8 B200 with the defaults (623 train / 108 val prompts, 16 x 8 samples per step):
# eval pass 0.68 -> 0.85 by step 2; 432-648 s per step, peak 68.4 GB allocated.
#
# The context-parallel configuration holds 262K-token sequences (CP8, 262K vLLM max_model_len, 262K
# micro-batches): rebuild with a larger --tokens for longer prompts, and keep prompt + MAX_RESPONSE_LENGTH
# within 262,144. run_dapo_fullft_262k_cp8_4node.sh is the full-FT variant.
#
# Context parallelism for qwen4_exp covers the QSA attention (split-K cuDNN QSA over CP chunks with an
# LSE merge), the PLE halo and the QSA indexer's pooled-key all-gather; GDN CP is megatron-core's.
# If the image puts its own cuDNN on LD_LIBRARY_PATH (e.g. /opt/cudnn), the Ray workers load those
# sublibraries instead of the pip cuDNN and QSA runs on that older version: drop that entry and set
# SKYRL_LD_LIBRARY_PATH_EXPORT=1 so the workers get the driver's path.
# Measured on 4 x 8 B200 at full length (4 prompts x 4 samples of ~255K tokens, 4,096-token
# responses): warm RL step ~200 s (generate 46 / logprobs 35 / train 104 / weight sync 10 s),
# peak 74.2 GB allocated.
#
# Everything else matches run_dapo_qwen3_8_flash_next_lora_sync_4node.sh (per-setting reasoning
# there and in run_gsm8k_qwen3_8_flash_next_lora_4node.sh).

# cuDNN QSA (see run_dapo_qwen3_8_flash_next_lora_sync_4node.sh). Context parallelism uses split-K
# cuDNN QSA, which is Blackwell-only: install the cudnn-frontend wheel for CP > 1.
QSA_CUDNN_WHEEL="${QSA_CUDNN_WHEEL:-}"

export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=5400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=128
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# Otherwise redirect_actor_output_to_file() swallows vLLM's errors.
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
# fla's TileLang GDN backward aborts on Blackwell; these recipes target B200, so default to Triton.
export FLA_TILELANG="${FLA_TILELANG:-0}"

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/Qwen3.8-Flash-Next}"
DATA_DIR="${DATA_DIR:-$HOME/data/keychain}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/keychain_64k_train_filtered.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/keychain_64k_val_filtered.parquet}"

NUM_NODES=4
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="${LOGGER:-wandb}"

MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-10}"

# Trainer parallelism. CP8 holds one 262K sequence per 8 GPUs (CP chunks are 32K tokens; CP must
# satisfy TP x CP <= 16, the GDN key-head count). TP is 1 or 2 (TP=2 runs with sequence parallelism). At TP=4 a rank has
# 6 query heads per KV head, which the cuDNN QSA kernels don't cover (FlexAttention fallback).
MEGATRON_TP="${MEGATRON_TP:-1}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-8}"
MEGATRON_EP="${MEGATRON_EP:-16}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"
if [ "$MEGATRON_CP" -gt 1 ] && [ -z "$QSA_CUDNN_WHEEL" ]; then
  echo "QSA_CUDNN_WHEEL must point at the cudnn-frontend wheel when MEGATRON_CP > 1 (split-K cuDNN QSA)." >&2
  exit 1
fi

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-257024}"
# The KeyChain filter was calibrated at 12,288-token responses.
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12288}"
INFERENCE_ENGINE_MAX_MODEL_LEN=262144        # prompt + response + chat-template headroom
OVERLONG_BUFFER_LEN=1024                     # penalty starts 1024 tokens before the cap
# Half-strength soft overlong penalty. At 1.0, full FT learned to cut its reasoning short: by step 20,
# held-out AIME accuracy on finished responses fell to 0.86 (0.99 at 0.5) with 33/240 finished-wrong.
OVERLONG_BUFFER_PENALTY_FACTOR="${OVERLONG_BUFFER_PENALTY_FACTOR:-0.5}"

# dp = 32 / (TP * PP * CP), and (mini_batch * n_samples) % dp == 0.
# dp = 4 here: 16 prompts x 8 samples = 128 sequences per step, 32 per DP rank, one optimizer step.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
EVAL_N_SAMPLES_PER_PROMPT=4
EVAL_INTERVAL="${EVAL_INTERVAL:-2}"
MAX_TOKENS_PER_MICROBATCH=262144  # must hold one full sequence

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
# Eval samples like training: at top_p 0.7 the base model is already near its ceiling on finished answers.
EVAL_TOP_P=1.0
# LoRA needs a much larger LR than full FT: at 1e-5 / 3e-5 the per-step update was ~1e-4 nats/token
# and runs stayed flat; 3e-4 with alpha 32 learns (held-out AIME 0.45 -> 0.85 in 10 steps).
LR="${LR:-3e-4}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
# vLLM's custom all-reduce registers CUDA-graph buffers over CUDA IPC, which fails on sleep mode's cumem
# allocator (colocate_all=true, TP>1): every TP worker dies at engine start with
# custom_all_reduce.cuh:164 'invalid argument'.
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "disable_custom_all_reduce": true}'
ROUTER_INIT_KWARGS='{"policy": "round_robin", "queue_size": 8192, "queue_timeout_secs": 1800}'

RUN_NAME="${RUN_NAME:-qwen3_8_flash_next_dapo_sync_lora_r16_262k_cp8_ep16}"
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
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=8 \
  trainer.policy.model.lora.rank=16 \
  trainer.policy.model.lora.alpha=32 \
  trainer.policy.model.lora.sync_mode=memory \
  trainer.policy.megatron_config.lora_config.merge_lora=false \
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
  trainer.ckpt_interval=0 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.distributed_executor_backend="mp" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.max_num_seqs=32 \
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
  environment.env_class=search \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen3_8_flash_next_dapo" \
  trainer.run_name="$RUN_NAME" \
  "$@"
