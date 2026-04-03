#!/usr/bin/env bash
# Task-gen specific run for Qwen3.5-35B: calls common run with task-gen entrypoint
# and 35B-specific config (TP=2, flash_attn=false, 72K input, chunked lm_head).
#
# Usage (from SkyPilot YAML run block):
#   bash scripts/fleet-task-gen-35b-run.sh
#
# Required env vars: WANDB_API_KEY, FLEET_API_KEY
# SkyPilot env vars: SKYPILOT_NUM_GPUS_PER_NODE, SKYPILOT_NODE_IPS
set -euo pipefail

# Export RUN_NAME so task_gen_env can tag rollout dumps
export RUN_NAME="task_gen_35b_$(python3 -c 'import os; print(os.urandom(4).hex())')"

# Defaults for vars normally set by SkyPilot YAML envs block
export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export MODALITY="${MODALITY:-tool_use}"
export NUM_EPOCHS="${NUM_EPOCHS:-20}"
export MAX_TURNS="${MAX_TURNS:-10}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-72000}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-8}"
export JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-sonnet-4.5}"
export EVALUATOR_MODEL="${EVALUATOR_MODEL:-anthropic/claude-sonnet-4.5}"
export K_ROLLOUTS="${K_ROLLOUTS:-4}"
export ALPHA="${ALPHA:-1.0}"
export MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-20}"

: "${FLEET_API_KEY:?Set FLEET_API_KEY before running}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

# Optional: per-env dataset filtering via TASK_GEN_ENV_CLASSES env var
ENV_FILTER_ARGS=()
if [ -n "${TASK_GEN_ENV_CLASSES:-}" ]; then
  echo "=== env_filter: $TASK_GEN_ENV_CLASSES ==="
  ENV_FILTER_ARGS+=("data.env_filter=$TASK_GEN_ENV_CLASSES")
fi

# Task-gen GRPO training with 35B model
# --entrypoint: task-gen entrypoint (not main_fleet)
# --env-class: task_gen environment (not fleet_task)
# TP=2: 8 engines × 2 GPUs each across 2 nodes (16 GPUs total)
# flash_attn=false: SDPA to avoid Xid 31 in GatedDeltaNet with vLLM 0.18.0
# loss_chunk_size=4096: chunked lm_head to avoid OOM on 131K vocab
# --no-pytorch-alloc-conf: disables expandable_segments (conflicts with vLLM CuMemAllocator)
bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 \
  --entrypoint integrations.fleet.entrypoints.main_task_gen \
  --env-class task_gen -- \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3.5-35B-A3B" \
  trainer.flash_attn=false \
  trainer.loss_chunk_size=4096 \
  trainer.use_sample_packing=false \
  generator.inference_engine_tensor_parallel_size=2 \
  trainer.epochs=${NUM_EPOCHS} \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=12 \
  trainer.use_hybrid_env_sampling=true \
  trainer.min_samples_per_env=1 \
  trainer.policy_mini_batch_size=12 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_ckpts_to_keep=1 \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=0.95 \
  generator.sampling_params.top_p=0.95 \
  'generator.sampling_params.stop=["</tool_call>", "</task>"]' \
  generator.eval_sampling_params.temperature=0.95 \
  generator.eval_sampling_params.top_p=0.95 \
  'generator.eval_sampling_params.stop=["</tool_call>", "</task>"]' \
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.algorithm.use_kl_loss=true \
  generator.max_turns=$MAX_TURNS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.trajectory_timeout_seconds=1800 \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=3 \
  generator.enforce_eager=false \
  generator.gpu_memory_utilization=0.65 \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-task-gen" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/task_gen_35b" \
  trainer.dump_data_batch=true \
  ++environment.skyrl_gym.task_gen.max_turns=$MAX_TURNS \
  ++environment.skyrl_gym.task_gen.judge_model="$JUDGE_MODEL" \
  ++environment.skyrl_gym.task_gen.k_rollouts=$K_ROLLOUTS \
  ++environment.skyrl_gym.task_gen.alpha=$ALPHA \
  ++environment.skyrl_gym.task_gen.max_eval_steps=$MAX_EVAL_STEPS \
  ++environment.skyrl_gym.task_gen.evaluator_model="$EVALUATOR_MODEL" \
  ++environment.skyrl_gym.task_gen.eval_k_rollouts=8 \
  ++environment.skyrl_gym.task_gen.tool_call_reward_per_call=0.02 \
  "${ENV_FILTER_ARGS[@]}" \
  "$@"
