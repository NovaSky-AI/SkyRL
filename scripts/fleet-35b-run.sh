#!/usr/bin/env bash
# Single source of truth for Qwen3.5-35B-A3B GRPO training config.
# Called by the SkyPilot YAML and by fleet-research run.sh.
#
# Required env vars: FLEET_API_KEY, WANDB_API_KEY
# Optional: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for S3 checkpoints)
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL root (scripts/ is directly under repo root)

# Defaults for vars normally set by SkyPilot YAML envs block
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export DATA_VERSION="${DATA_VERSION:-v55}"
export MODALITY="${MODALITY:-tool_use}"
export NUM_EPOCHS="${NUM_EPOCHS:-20}"
export MAX_TURNS="${MAX_TURNS:-50}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-72000}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-8}"
export ENV_KEYS="${ENV_KEYS:-}"
export DIFFICULTY="${DIFFICULTY:-}"
export RUN_ID="${RUN_ID:-}"
export MAX_TASKS="${MAX_TASKS:-}"
export RESUME_RUN_NAME="${RESUME_RUN_NAME:-}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export S3_DATASET_BUCKET="${S3_DATASET_BUCKET:-fleet-internal-datasets}"
export S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:-skyrl-checkpoints}"
export S3_TRAJECTORY_BUCKET="${S3_TRAJECTORY_BUCKET:-skyrl-trajectories}"

: "${FLEET_API_KEY:?Set FLEET_API_KEY before running}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 -- \
  environment.skyrl_gym.fleet_task.ttl_seconds=900 \
  environment.skyrl_gym.fleet_task.partial_reward=true \
  environment.skyrl_gym.fleet_task.enable_hints=true \
  environment.skyrl_gym.fleet_task.n_hint_samples=2 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="Qwen/Qwen3.5-35B-A3B" \
  trainer.flash_attn=true \
  trainer.loss_chunk_size=4096 \
  trainer.use_sample_packing=false \
  +generator.chat_template_kwargs='{enable_thinking:true}' \
  generator.inference_engine_tensor_parallel_size=2 \
  trainer.epochs=${NUM_EPOCHS} \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=false \
  trainer.eval_interval=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.use_hybrid_env_sampling=true \
  trainer.min_samples_per_env=1 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_ckpts_to_keep=1 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=0.9 \
  generator.sampling_params.top_p=0.95 \
  'generator.sampling_params.stop=["</tool_call>"]' \
  'generator.eval_sampling_params.stop=["</tool_call>"]' \
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.algorithm.use_kl_loss=true \
  generator.max_turns=$MAX_TURNS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=3 \
  generator.enforce_eager=false \
  generator.gpu_memory_utilization=0.65 \
  generator.inject_context_status=true \
  generator.context_warning_threshold=0.90 \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-tool-use-grpo" \
  trainer.run_name="fleet_qwen35_35b_${MODALITY}_${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/fleet_qwen35_35b_${MODALITY}" \
  trainer.export_path="$HOME/exports" \
  trainer.dump_data_batch=true \
  "$@"
