set -x

# Colocated DAPO training+generation for Qwen3.5-2B (dense) on DAPO with Megatron,
# WITH Multi-Token Prediction (MTP) speculative decoding for faster rollout.
#
# This is the spec-decode counterpart of run_megatron_dapo_qwen3.5_2b.sh. Every
# knob below is IDENTICAL to the no-spec script (same batch sizes, LR, parallelism,
# sampling) so a reward-curve / throughput comparison is apples-to-apples. The ONLY
# difference is the `trainer.mtp.*` block at the bottom.
#
# What MTP on does (single high-level `trainer.mtp` knob, see skyrl/train/config/config.py):
#   - Training side: builds + trains Qwen3.5-2B's native MTP head (the model ships
#     `mtp_num_hidden_layers: 1`) with a decoupled draft loss (soft-CE distillation against
#     the policy's own detached next-token distribution). The draft gradient never pulls on
#     the policy backbone.
#   - Inference side: enables vLLM MTP speculative decoding
#     (`speculative_config={"method": "mtp", "num_speculative_tokens": 1}`). vLLM loads the
#     MTP head from the same policy checkpoint, and SkyRL's weight sync keeps the draft head
#     in sync with the trained policy each step.
#   - The per-step draft acceptance rate is logged as `vllm/draft_acceptance_rate`.
#
# Runs on 1 node of 8xH100s (80GB each).
#
# Prepare data onto the fast local disk first:
#   DATA_DIR=/mnt/local_storage/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# Then launch:
#   bash examples/train/megatron/run_megatron_dapo_qwen3.5_2b_specdecode.sh

MODEL_NAME="Qwen/Qwen3.5-2B"
# Use the fast, non-persistent local disk for data (not the ~/default quota).
DATA_DIR="/mnt/local_storage/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
NUM_NODES=1
NUM_GPUS_PER_NODE=8
# 2B is small: TP=1 inference per engine (no TP comm), one engine per GPU.
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1
LOGGER="wandb"  # change to "console" to print to stdout

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
# IDENTICAL to the no-spec run (8192) for an apples-to-apples comparison. The MTP draft loss adds a
# full [seq, vocab] softmax tensor on top of the main lm-head gradient, but this fits at TP=2 with the
# default allocator now that the O(num_microbatches) `mtp_student_logits` accumulation is fixed.
MAX_RESPONSE_LENGTH=$((1024 * 8))

# repro run parameters
# Quality-comparison profile: larger batch lowers reward-curve variance (~4x vs batch 32)
# so a spec-decode quality effect is detectable. rollout = 128 * 8 = 1024 seqs.
# MINI_BATCH_SIZE == TRAIN_BATCH_SIZE => 1 on-policy update/rollout. Keep IDENTICAL across no-spec/spec runs.
TRAIN_BATCH_SIZE=128
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=16
ENFORCE_EAGER=true # cuda graphs can cause some instability
LR=1e-6

# megatron config -- Qwen3.5-2B is a dense model, so no expert parallelism.
# TP=2 (matching the no-spec run): the decoupled MTP draft loss materializes a full
# [batch, seq, vocab] float32 softmax tensor on top of the main lm-head gradient at Qwen3.5's 248K
# vocab, but this fits at TP=2 with the default allocator now that the O(num_microbatches)
# `mtp_student_logits` accumulation is fixed. NOTE: raising TP does NOT meaningfully shrink the
# training footprint (it's activation/optimizer-bound, not vocab-bound), and TP=8 also triggered a
# CUDA-IPC weight-sync failure (pidfd_getfd EPERM) when broadcasting the 8-way-sharded policy to the
# TP=1 vLLM engines. So TP=2 is both the apples-to-apples match and the known-good weight-sync config.
# TP=2, PP=1, CP=1 => DP=4.
MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=null


# TIS parameters
TIS_IMP_RATIO_CAP=2.0
TIS_TYPE=token


# Multi-Token Prediction (MTP) speculative decoding.
# Qwen3.5-2B ships 1 native MTP head (`mtp_num_hidden_layers: 1`); training always trains the
# checkpoint's heads. NUM_SPECULATIVE_TOKENS is the vLLM *draft depth* only — values > 1 reuse the
# single head autoregressively at draft time (more speedup, but per-position acceptance decays with
# depth since the head never trains on its own outputs). Try 2-3 for extra acceleration.
MTP_ENABLED=true
MTP_NUM_SPECULATIVE_TOKENS=1
MTP_LOSS_TYPE="soft_ce" # "soft_ce" (distill against policy) | "hard_ce" (ground-truth next tokens)
MTP_LOSS_WEIGHT=0.1
# NOTE: trainer.policy.megatron_config.mtp_detach_shared_output now defaults to true: the draft loss
# trains ONLY the MTP-head params; the tied embedding/lm_head is detached (output projection AND the
# MTP block's re-embedding), so the draft gradient no longer nudges the policy's own logits. Set it
# to false for the old NeMo-RL behaviour (shared head also trained by the draft loss).
# Top-k draft loss: distill only the teacher's top-k tokens instead of the full 248K vocab, keeping
# draft-loss memory at O(seq*k) vs O(seq*vocab). This is now an optional throughput knob, NOT a
# memory requirement: the full-vocab soft-CE fits comfortably at TP=2 since the real OOM cause (the
# O(num_microbatches) accumulation of `mtp_student_logits`) was fixed by freeing each microbatch's
# student logits right after its draft backward. No PYTORCH_CUDA_ALLOC_CONF tuning is needed.
# null => exact full-vocab soft-CE. Set to e.g. 64 to use the memory-light top-k approximation.
MTP_LOSS_TOPK=null


# Qwen3.5 flags
REMOVE_MICROBATCH_PADDING=false # sample packing is not yet supported for GDN layers in megatron - see: https://github.com/NVIDIA/Megatron-LM/pull/2644
ENGINE_INIT_KWARGS='{"gdn_prefill_backend": "triton"}' # see https://github.com/vllm-project/vllm/issues/36921#issuecomment-4109702738
DISTRIBUTED_EXECUTOR_BACKEND="mp"
export _SKYRL_USE_NEW_INFERENCE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF here (neither expandable_segments:True nor max_split_size_mb).
# - expandable_segments:True makes PyTorch allocate via virtual-memory segments incompatible with the
#   legacy CUDA-IPC handle used by colocated weight sync -> it falls back to pidfd_getfd, which this
#   cluster's ptrace_scope blocks (pidfd_getfd: Operation not permitted).
# - max_split_size_mb over-reserves PyTorch memory and starves NCCL's external cudaMalloc at grad-sync
#   ("Failed to CUDA calloc ... bytes" in reduce_scatter).
# Neither is needed: the draft-loss OOM was an O(num_microbatches) accumulation of `mtp_student_logits`,
# now fixed by freeing each microbatch's student logits right after its draft backward. The full-vocab
# soft-CE fits at TP=2 with the default allocator.

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
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
  generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
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
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.epochs=10 \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=5 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.mtp.enabled=$MTP_ENABLED \
  trainer.mtp.num_speculative_tokens=$MTP_NUM_SPECULATIVE_TOKENS \
  trainer.mtp.loss_type=$MTP_LOSS_TYPE \
  trainer.mtp.loss_weight=$MTP_LOSS_WEIGHT \
  trainer.policy.megatron_config.mtp_loss_topk=$MTP_LOSS_TOPK \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen3_5_dapo_sd" \
  trainer.run_name="sd_dapo_qwen3_5_2b_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  trainer.export_path="/mnt/local_storage/exports/sd_dapo_qwen3_5_2b_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="/mnt/local_storage/ckpts/sd_dapo_qwen3_5_2b_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  $@
