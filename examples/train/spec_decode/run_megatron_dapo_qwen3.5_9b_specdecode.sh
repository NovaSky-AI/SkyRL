set -x

# Colocated DAPO training+generation for Qwen3.5-9B (dense) on DAPO data with Megatron,
# with Multi-Token Prediction (MTP) speculative decoding for faster rollout. Runs on 1x8 H100.
# DATA_DIR=/mnt/local_storage/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/spec_decode/run_megatron_dapo_qwen3.5_9b_specdecode.sh

MODEL_NAME="Qwen/Qwen3.5-9B"
# Use the fast, non-persistent local disk for data (not the ~/default quota).
DATA_DIR="/mnt/local_storage/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
NUM_NODES=1
NUM_GPUS_PER_NODE=8
# 9B is ~4.5x the 2B: a single full-weight copy is ~18GB in bf16. Use inference
# TP=2 (4 engines) so each engine's weights are halved (~9GB/GPU) and there is
# more headroom for KV cache during generation. TP=2 comm stays on NVLink.
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
MAX_RESPONSE_LENGTH=$((1024 * 8))

# repro run parameters
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=16
ENFORCE_EAGER=true # cuda graphs can cause some instability
LR=1e-6

# megatron config -- Qwen3.5-9B is a dense model, so no expert parallelism.
# TP=4 (up from 2 on the 2B): 9B params + Adam states + 8K-token activations
# need more sharding to fit at micro batch 1. TP>1 auto-enables sequence
# parallelism, sharding activations/vocab-logits across the TP group.
# TP=4, PP=1, CP=1 => DP=2. TP stays within the single-node NVLink domain.
MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=null

# optimizer offload -- OFF by default (with colocate_all the inference engine is
# offloaded during training, so the full 80GB is available for the optimizer).
# Flip to true if you hit OOM in the optimizer step / grad sync.
OPTIMIZER_OFFLOAD=false
OPTIMIZER_OFFLOAD_FRACTION=1.0

# TIS parameters
TIS_IMP_RATIO_CAP=2.0
TIS_TYPE=token


# Multi-Token Prediction (MTP) speculative decoding.
# Qwen3.5-9B ships 1 native MTP head (`mtp_num_hidden_layers: 1`); training always trains the
# checkpoint's heads. NUM_SPECULATIVE_TOKENS is the vLLM *draft depth* only — values > 1 reuse the
# single head autoregressively at draft time (more speedup, but per-position acceptance decays with
# depth since the head never trains on its own outputs). Here k=3.
MTP_ENABLED=true
MTP_NUM_SPECULATIVE_TOKENS=3
MTP_LOSS_TYPE="soft_ce" # "soft_ce" (distill against policy) | "hard_ce" (ground-truth next tokens)
MTP_LOSS_WEIGHT=1.0
# NOTE: trainer.policy.megatron_config.mtp_detach_shared_output now defaults to true: the draft loss
# trains ONLY the MTP-head params; the tied embedding/lm_head is detached (output projection AND the
# MTP block's re-embedding), so the draft gradient no longer nudges the policy's own logits. Set it
# to false for the old NeMo-RL behaviour (shared head also trained by the draft loss).
# Top-k draft loss: distill only the teacher's top-k tokens instead of the full 248K vocab, keeping
# draft-loss memory at O(seq*k) vs O(seq*vocab). Only applies to mtp_loss_type="soft_ce". Here k=256.
MTP_LOSS_TOPK=256


# Qwen3.5 flags
REMOVE_MICROBATCH_PADDING=false # sample packing is not yet supported for GDN layers in megatron - see: https://github.com/NVIDIA/Megatron-LM/pull/2644
ENGINE_INIT_KWARGS='{"gdn_prefill_backend": "triton"}' # see https://github.com/vllm-project/vllm/issues/36921#issuecomment-4109702738
DISTRIBUTED_EXECUTOR_BACKEND="mp"
export _SKYRL_USE_NEW_INFERENCE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

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
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
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
  trainer.ckpt_interval=50 \
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
  trainer.project_name="qwen3_5_dapo_2" \
  trainer.run_name="sd_dapo_qwen3_5_9b_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  trainer.export_path="/mnt/local_storage/exports/sd_dapo_qwen3_5_9b_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="/mnt/local_storage/ckpts/sd_dapo_qwen3_5_9b_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  $@
