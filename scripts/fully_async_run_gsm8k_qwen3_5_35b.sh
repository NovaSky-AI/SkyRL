#!/usr/bin/env bash
# Fully async GRPO training+generation for Qwen3.5-35B-A3B on GSM8K.
# Routes through fleet-common-run.sh for multi-node Ray cluster setup.
set -euo pipefail
cd "$(dirname "$0")/.."

export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export MODALITY="${MODALITY:-gsm8k}"

: "${NUM_INFERENCE_GPUS:=${NUM_INFERENCE_ENGINES:-4}}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

# --- Prepare GSM8K dataset if not already present ---
GSM8K_DATA_DIR="$HOME/data/fleet/gsm8k"
if [ ! -f "$GSM8K_DATA_DIR/train.parquet" ]; then
  echo "Preparing GSM8K dataset at $GSM8K_DATA_DIR..."
  source .venv/bin/activate
  uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir "$GSM8K_DATA_DIR"
fi

# Fully async specific configuration knobs:
: "${MINI_BATCH_SIZE:=16}"
: "${MAX_STALENESS_STEPS:=4}"
: "${NUM_PARALLEL_GENERATION_WORKERS:=$(( MINI_BATCH_SIZE * (MAX_STALENESS_STEPS + 1) ))}"

TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

RUN_NAME=gsm8k-fully-async-qwen3.5_35B-useTIS_${TIS_TYPE}-maxStale${MAX_STALENESS_STEPS}-numCon${NUM_PARALLEL_GENERATION_WORKERS}

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 \
  --entrypoint examples.train.fully_async.main_fully_async \
  --env-class gsm8k \
  --data-dir-name gsm8k \
  -- \
  trainer.fully_async.max_staleness_steps=${MAX_STALENESS_STEPS} \
  trainer.fully_async.num_parallel_generation_workers=${NUM_PARALLEL_GENERATION_WORKERS} \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.policy.model.path="Qwen/Qwen3.5-35B-A3B" \
  trainer.flash_attn=false \
  trainer.loss_chunk_size=4096 \
  trainer.use_sample_packing=false \
  +generator.chat_template_kwargs='{enable_thinking:true}' \
  trainer.placement.colocate_all=false \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_nodes=1 \
  generator.inference_engine.num_engines=$NUM_INFERENCE_GPUS \
  generator.inference_engine.tensor_parallel_size=2 \
  trainer.epochs=20 \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=false \
  trainer.eval_interval=4 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${MINI_BATCH_SIZE} \
  trainer.policy_mini_batch_size=${MINI_BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.65 \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-tool-use-grpo" \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  generator.inference_engine.enforce_eager=false \
  "$@"
