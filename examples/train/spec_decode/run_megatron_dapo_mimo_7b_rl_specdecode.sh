set -x

# Colocated DAPO training+generation for XiaomiMiMo/MiMo-7B-RL (dense) on DAPO with
# Megatron, WITH native Multi-Token Prediction (MTP) speculative decoding for faster rollout.
#
# This is the MiMo-7B-RL counterpart of run_megatron_dapo_qwen3.5_9b_specdecode.sh. The
# DAPO knobs (batch sizes, LR, sampling, clipping) are kept identical to the Qwen recipe so the
# reward-curve / throughput comparison stays apples-to-apples; only the model and the few
# architecture-specific flags below differ.
#
# Why MiMo-7B-RL is a good MTP target:
#   - MiMo was pretrained WITH MTP. The released checkpoint ships one native MTP module
#     (`num_nextn_predict_layers: 1`, the DeepSeek-style key), so both the training-side head and
#     the inference-side drafter come from the same pretrained weights — no cold-start head.
#   - The backbone is a plain Qwen2-style dense transformer (36 layers, full attention,
#     `attention_bias: true`), so there are NO GDN / linear-attention layers. Two consequences vs
#     the Qwen3.5 recipe: (a) megatron sample packing IS supported, so we set
#     remove_microbatch_padding=true for better throughput; (b) we drop the `gdn_prefill_backend`
#     vLLM engine kwarg — it does nothing here.
#   - Embeddings are UNTIED (`tie_word_embeddings: false`). The decoupled draft loss therefore
#     isolates the output projection by detaching `output_layer.weight` (the untied branch of
#     mtp_detach_shared_output, on by default), so the draft gradient trains ONLY the MTP-head params.
#
# What MTP on does (single high-level `trainer.mtp` knob, see skyrl/train/config/config.py):
#   - Training side: megatron-bridge's MiMo bridge builds MiMo-7B-RL's native MTP head (from
#     `num_nextn_predict_layers`) and trains it with a decoupled draft loss (top-k soft-CE
#     distillation against the policy's own detached next-token distribution). The draft gradient
#     never pulls on the policy backbone.
#   - Inference side: enables vLLM MTP speculative decoding. The generic `{"method": "mtp"}` config
#     auto-detects MiMo's `mimo_mtp` drafter from the model config. With a single trained head,
#     depth>1 reuses that head autoregressively at draft time. SkyRL's weight sync keeps the draft
#     head in sync with the trained policy each step.
#   - The per-step draft acceptance rate is logged as `vllm/draft_acceptance_rate`.
#
# Runs on 1 node of 8xH100s (80GB each).
#
# Prepare data onto the fast local disk first:
#   DATA_DIR=/mnt/local_storage/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# Then launch:
#   bash examples/train/megatron/run_megatron_dapo_mimo_7b_rl_specdecode.sh

MODEL_NAME="XiaomiMiMo/MiMo-7B-RL"
# Use the fast, non-persistent local disk for data (not the ~/default quota).
DATA_DIR="/mnt/local_storage/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
NUM_NODES=1
NUM_GPUS_PER_NODE=8
# One vLLM engine per GPU (TP=1). A single MiMo-7B copy is ~14GB in bf16, which fits
# comfortably per engine alongside KV cache + the small MTP drafter at gpu_memory_utilization=0.5
# (the policy is offloaded during generation under colocate_all).
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
# MiMo-7B-RL has max_position_embeddings=32768, so 2K prompt + 8K response fits with room to spare.

# repro run parameters
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=16
ENFORCE_EAGER=true # cuda graphs can cause some instability
LR=1e-6

# megatron config -- MiMo-7B-RL is a dense model, so no expert parallelism.
# TP=4: 7B params + Adam states + 8K-token activations fit at micro batch 1 with headroom.
# TP>1 auto-enables sequence parallelism, sharding activations/vocab-logits across the TP group.
# TP=4, PP=1, CP=1 => DP=2. TP stays within the single-node NVLink domain. (7B is smaller than the
# Qwen3.5-9B recipe, so TP=2 => DP=4 likely also fits and gives more throughput -- try it if you
# have headroom.)
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
# MiMo-7B-RL ships 1 native MTP head (`num_nextn_predict_layers: 1`); training always trains the
# checkpoint's heads (mtp_num_layers left at its default => inferred from the model's HF config).
# NUM_SPECULATIVE_TOKENS is the vLLM *draft depth* only — values > 1 reuse the single head
# autoregressively at draft time (more speedup, but per-position acceptance decays with depth since
# the head never trains on its own outputs). Here k=3.
MTP_ENABLED=true
MTP_NUM_SPECULATIVE_TOKENS=3
MTP_LOSS_TYPE="soft_ce" # "soft_ce" (distill against policy) | "hard_ce" (ground-truth next tokens)
MTP_LOSS_WEIGHT=0.5
# NOTE: trainer.policy.megatron_config.mtp_detach_shared_output defaults to true. MiMo has UNTIED
# embeddings, so this detaches the (separate) output_layer.weight passed into the draft projection,
# along with the MTP block's re-embedding, so the draft gradient trains ONLY the MTP-head params and
# never nudges the policy's own logits. Set it to false for the old NeMo-RL behaviour (shared head
# also trained by the draft loss).
# Top-k draft loss: distill only the teacher's top-k tokens instead of the full vocab, keeping
# draft-loss memory at O(seq*k) vs O(seq*vocab). Only applies to mtp_loss_type="soft_ce". Here k=256.
MTP_LOSS_TOPK=256


# MiMo flags -- plain Qwen2-style dense attention (no GDN), so sample packing is supported and no
# special vLLM prefill backend is needed.
REMOVE_MICROBATCH_PADDING=true
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
  trainer.project_name="mimo_7b_rl_dapo" \
  trainer.run_name="sd_dapo_mimo_7b_rl_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  trainer.export_path="/mnt/local_storage/exports/sd_dapo_mimo_7b_rl_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="/mnt/local_storage/ckpts/sd_dapo_mimo_7b_rl_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}" \
  $@
