#!/usr/bin/env bash
set -euo pipefail
set -x

# Full RL training run: VLM RL on VisGym maze_2d/easy with mixed_qwen3vl SFT checkpoint.
#
# Colocated setup: 8 GPUs shared between vLLM inference and FSDP training.
# Uses the VLM generator (multi-modal) via custom entrypoint.
#
# Prerequisites – generate the stub dataset:
#   uv run examples/train/visgym/visgym_dataset.py \
#       --env_id maze_2d/easy --num_rows 256 \
#       --output_dir ~/data/visgym_maze_2d_easy
#
# Usage:
#   bash examples/train/visgym/run_visgym_full_mazed2d_easy-finalized.sh

: "${ENV_ID:=maze_2d/easy}"
: "${DATA_DIR:="$HOME/data/visgym_maze_2d_easy"}"
: "${NUM_INFERENCE_GPUS:=8}"
: "${NUM_TRAIN_GPUS:=8}"
: "${LOGGER:=wandb}"

: "${EXPORT_PATH:="$HOME/exports/visgym_maze_2d_easy_format_reward_0.2"}"
: "${DUMP_TRAINING_BATCHES:=false}"
: "${DUMP_EVAL_RESULTS:=true}"
: "${EVAL_INTERVAL:=10}"

: "${TRAIN_BATCH_SIZE:=32}"
: "${POLICY_MINI_BATCH_SIZE:=16}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${NUM_DATASET_ROWS:=256}"
: "${EVAL_DATA_DIR:="$HOME/data/visgym_maze_2d_easy_eval"}"

# ── Generate stub dataset if it doesn't exist ────────────────────────
if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "=== Generating train stub dataset for $ENV_ID ==="
  uv run examples/train/visgym/visgym_dataset.py \
    --env_id "$ENV_ID" \
    --num_rows "$NUM_DATASET_ROWS" \
    --output_dir "$DATA_DIR"
fi

if [ ! -f "$EVAL_DATA_DIR/train.parquet" ]; then
  echo "=== Generating eval stub dataset for $ENV_ID ==="
  uv run examples/train/visgym/visgym_dataset.py \
    --env_id "$ENV_ID" \
    --num_rows 64 \
    --seed \
    --output_dir "$EVAL_DATA_DIR"
fi

MODEL_PATH="/mnt/local_storage/visgym_model_repo/mixed_qwen3vl"
_SKYRL_USE_NEW_INFERENCE=1 \
uv run --isolated --extra fsdp \
  python examples/train/visgym/visgym_entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$EVAL_DATA_DIR/train.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_TRAIN_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_TRAIN_GPUS \
  trainer.ref.fsdp_config.cpu_offload=false \
  generator.inference_engine.num_engines=$NUM_INFERENCE_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.engine_init_kwargs.max_model_len=60000 \
  trainer.epochs=20 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=0.7 \
  generator.max_turns=15 \
  generator.max_input_length=8192 \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.is_vlm=true \
  generator.batched=false \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  environment.env_class=visgym \
  trainer.logger="$LOGGER" \
  trainer.project_name="vlm_maze_2d_easy" \
  trainer.run_name="maze_2d_easy_full_format_reward_0.2-reduced_bsz" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_data_batch="$DUMP_TRAINING_BATCHES" \
  trainer.dump_eval_results="$DUMP_EVAL_RESULTS" \
  trainer.ckpt_path="$HOME/ckpts/visgym_maze_2d_easy_format_reward_0.2" \
  trainer.use_sample_packing=false \
  trainer.eval_interval="$EVAL_INTERVAL" \
  trainer.ckpt_interval=10 \
  "$@"
