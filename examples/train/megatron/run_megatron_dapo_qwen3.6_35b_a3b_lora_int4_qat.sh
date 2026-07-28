set -x

# DAPO LoRA training for Qwen3.6-35B-A3B on ONE 8xH200 node, colocated, with the
# INT4 fake-quant QAT stack. vLLM serves the model as compressed-tensors INT4
# (W4A16); the Megatron trainer holds BF16 masters and (when QAT is on)
# fake-quantizes the MoE expert GEMMs onto the same INT4 grid in the forward pass
# (STE backward), with TIS correcting the residual train/infer mismatch.
#
# First download data:
# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# 
# Then run the script:
# Two ablation modes (QAT_MODE), both serving INT4 in vLLM so the comparison
# isolates the effect of fake-quant + TIS:
#   QAT_MODE=off : fake-quant OFF + TIS OFF  -> uncorrected BF16(train)/INT4(infer) mismatch
#   QAT_MODE=on  : fake-quant ON  + TIS ON   -> corrected (on-policy)
#
#   QAT_MODE=off bash examples/train/megatron/run_megatron_dapo_qwen3.6_35b_a3b_lora_int4_qat.sh
#   QAT_MODE=on  bash examples/train/megatron/run_megatron_dapo_qwen3.6_35b_a3b_lora_int4_qat.sh

# INT4 actor served by vLLM; BF16 masters loaded by the trainer (Megatron-Bridge
# can't load compressed-tensors, so it reads BF16 from FAKE_QUANT_BF16_PATH).
MODEL_NAME="${MODEL_NAME:-casperhansen/Qwen3.6-35B-A3B-INT4-RTN}"
FAKE_QUANT_BF16_PATH="${FAKE_QUANT_BF16_PATH:-Qwen/Qwen3.6-35B-A3B}"

DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"

# --- ONE 8xH200 node, colocated. num_policy_gpus (8) == num_rollout_gpus (1*8). ---
NUM_NODES=1
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8
LOGGER="wandb"

# --- QAT / TIS ablation toggle ---
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
# Keep overlong (truncated) responses so the batch is never empty after filtering
# (Qwen3.6 math CoT often exceeds the short response cap used for a quick run).
APPLY_OVERLONG_FILTERING=false
OVERLONG_BUFFER_LEN=$((1024 * 1))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

USE_KL_LOSS=false
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 4))

# --- reduced scale so the OFF vs ON comparison is quick to eyeball in wandb ---
TRAIN_BATCH_SIZE=16
MINI_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=8
EVAL_N_SAMPLES_PER_PROMPT=8
ENFORCE_EAGER=false
LR=1e-5

LORA_RANK=32
LORA_ALPHA=32

# megatron config (8 GPUs: TP=4, EP=8/ETP=1 -> DP=2)
MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

TIS_IMP_RATIO_CAP=2.0

OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Qwen3.6 flags
LANGUAGE_MODEL_ONLY=True
ENGINE_INIT_KWARGS='{"gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}'
DISTRIBUTED_EXECUTOR_BACKEND="mp"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# On Blackwell, use the following env vars:
# export VLLM_USE_FLASHINFER_MOE_FP16=0   # force triton moe backend since flashinfer trtllm bf16 MoE kernel requires expert intermediate_size to be a multiple of 128
# export FLA_TILELANG=0   # force triton gdn backend since fla's default TileLang GDN backend aborts in the packed backward. leave unset on hopper, since Triton GDN backward is broken there: https://github.com/fla-org/flash-linear-attention/issues/640#issuecomment-4236520788

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
  trainer.policy.model.fake_int4_qat.enabled=$FAKE_QUANT_ENABLED \
  trainer.policy.model.fake_int4_qat.group_size=32 \
  trainer.policy.model.fake_int4_qat.scale_divisor=7.5 \
  trainer.policy.model.fake_int4_qat.bf16_base_path="$FAKE_QUANT_BF16_PATH" \
  trainer.policy.megatron_config.lora_config.merge_lora=false \
  trainer.fused_lm_head_logprob=true \
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
  generator.inference_engine.gpu_memory_utilization=0.6 \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen3_6_dapo_lora_int4qat" \
  trainer.run_name="dapo_lora_r32_qwen3_6_35b_a3b_1node_${RUN_SUFFIX}" \
  trainer.export_path="$HOME/exports/dapo_lora_qwen3_6_${RUN_SUFFIX}" \
  trainer.hf_save_interval=0 \
  trainer.resume_mode=none \
  trainer.max_ckpts_to_keep=1 \
  trainer.ckpt_path="$HOME/ckpts/dapo_lora_qwen3_6_${RUN_SUFFIX}" \
  $@
