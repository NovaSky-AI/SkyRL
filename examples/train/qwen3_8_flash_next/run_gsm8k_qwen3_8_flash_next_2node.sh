set -x

# Colocated GRPO for Qwen3.8-Flash-Next (HF qwen4_exp, ~180B MoE) on GSM8K with Megatron.
# Two nodes of 8xH100-80G (16 GPUs).
#
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   export WANDB_API_KEY=<your_key_here>   # or leave LOGGER=console
#   MODEL_PATH=/path/to/Qwen3.8-Flash-Next bash examples/train/qwen3_8_flash_next/run_gsm8k_qwen3_8_flash_next_2node.sh
#   MODE=fullft bash ...   # full fine-tune (optimizer state on CPU), default MODE=lora
#
# Put MODEL_PATH on node-local disk (same path on every node): every vLLM TP rank and every
# trainer rank reads the 360 GB checkpoint at startup.
#
# Megatron support is SkyRL's own (skyrl/backends/skyrl_train/patches/megatron/qwen4_exp), which
# constrains the recipe:
#   - TP=CP=PP=1; scale is expert parallelism. EP=16 also shards the frozen ~51B-parameter PLE
#     n-gram table by head (16 heads -> one per rank, ~6.4 GB each).
#   - Text only (language_model_only) and packed sequences (remove_microbatch_padding).
#   - Sequences must stay within the QSA indexer budget (2048 + 3 tokens), where the indexer
#     selects every key and attention is exactly dense; longer sequences raise. Hence the
#     512 + 1024 length caps.
#   - MTP is not trained (mtp_num_layers=0); vLLM's drafter keeps the base MTP weights, so do
#     not enable MTP speculative decoding for long runs.

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/Qwen3.8-Flash-Next}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
MODE="${MODE:-lora}"
LOGGER="${LOGGER:-console}"  # change to "wandb" to log to wandb

NUM_NODES=2
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
MAX_TOKENS_PER_MICROBATCH=8192

MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=16
MEGATRON_ETP=1

if [ "$MODE" = "lora" ]; then
  # Frozen bf16 base ~31 GB/GPU (dense 10 + experts 15 + PLE shard 6.4); adapters on all linears.
  LR=1e-5
  MODE_ARGS=(
    trainer.policy.model.lora.rank=32
    trainer.policy.model.lora.alpha=32
    trainer.policy.megatron_config.lora_config.merge_lora=true
  )
else
  # Full FT: ~173B trainable params (PLE table and QSA indexer stay frozen). Expert DP is 1 at
  # EP=16, so each rank holds ~7.5B expert params' fp32 master + Adam moments (~90 GB): the
  # optimizer must live on CPU (~720 GB per node).
  LR=1e-6
  MODE_ARGS=()
fi

ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"'}'
RUN_NAME="qwen3_8_flash_next_gsm8k_${MODE}_ep${MEGATRON_EP}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.strategy=megatron \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.loss_reduction="sequence_mean" \
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
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="full" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method="uniform" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.megatron_config.transformer_config_kwargs.mlp_chunks_for_training=8 \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.remove_microbatch_padding=true \
  trainer.use_expandable_segments=true \
  trainer.fused_lm_head_logprob=true \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/$RUN_NAME" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.distributed_executor_backend="mp" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen3_8_flash_next_gsm8k" \
  trainer.run_name="$RUN_NAME" \
  "${MODE_ARGS[@]}" \
  "$@"
