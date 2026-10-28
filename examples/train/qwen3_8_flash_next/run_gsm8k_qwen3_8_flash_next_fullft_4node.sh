set -x

# Colocated GRPO for Qwen3.8-Flash-Next (HF qwen4_exp, ~180B MoE) on GSM8K with Megatron, full
# fine-tuning. Four nodes of 8xB200 (32 GPUs, ~178 GiB usable each).
# LoRA: run_gsm8k_qwen3_8_flash_next_lora_4node.sh (constraints listed there).
#
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   export WANDB_API_KEY=<your_key_here>   # or leave LOGGER=console
#   MODEL_PATH=/path/to/Qwen3.8-Flash-Next bash examples/train/qwen3_8_flash_next/run_gsm8k_qwen3_8_flash_next_fullft_4node.sh
#
# Memory (per GPU): EP16 over 32 GPUs gives expert-DP 2, so the distributed optimizer shards the
# expert state over 2 ranks and the dense state over 32. The optimizer stays on the GPU (no CPU
# offload): bf16 weights ~25 GB + bf16 grads ~25 GB (grad_reduce_in_fp32=false) + FP32 masters
# and Adam moments. Colocated with vLLM: SkyRL offloads the trainer's weights, grads and optimizer
# state to the host while vLLM generates. The weight sync needs vLLM's TP8 weights (~40 GiB) awake
# next to the trainer weights and the largest fused expert tensor's receive buffer (~3.1 GiB).

MODEL_PATH="${MODEL_PATH:-/mnt/local_storage/Qwen3.8-Flash-Next}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
LOGGER="${LOGGER:-console}"  # change to "wandb" to log to wandb

NUM_NODES=4
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=8

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
MAX_TOKENS_PER_MICROBATCH=4096
# vLLM sizes its KV pool at engine start; the trainer's post-offload residual grows to ~24 GiB/GPU
# after the first optimizer step, and 0.85 OOMed adding the adapter at step 2. At 0.6 vLLM has
# ~950 GDN state blocks, one per running sequence, so max_num_seqs is capped below the 1024 default.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"

# Trainer parallelism. TP is 1 or 2 (TP=2 runs with sequence parallelism). At TP=4 a rank has
# 6 query heads per KV head, which the cuDNN QSA kernels don't cover (FlexAttention fallback).
MEGATRON_TP="${MEGATRON_TP:-1}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_EP="${MEGATRON_EP:-16}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"

LR=1e-6

# vLLM's custom all-reduce registers CUDA-graph buffers over CUDA IPC, which fails on sleep mode's cumem
# allocator (colocate_all=true, TP>1): every TP worker dies at engine start with
# custom_all_reduce.cuh:164 'invalid argument'.
ENGINE_INIT_KWARGS='{"max_model_len": '"$INFERENCE_ENGINE_MAX_MODEL_LEN"', "disable_custom_all_reduce": true}'
RUN_NAME="qwen3_8_flash_next_gsm8k_fullft_ep${MEGATRON_EP}"

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
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=false \
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
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  generator.inference_engine.max_num_seqs=512 \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.batched=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen3_8_flash_next_gsm8k" \
  trainer.run_name="$RUN_NAME" \
  "$@"
