#!/usr/bin/env bash
# Task-gen specific run: calls common run with task-gen entrypoint and hydra overrides
#
# Usage (from SkyPilot YAML run block):
#   bash skyrl-train/scripts/fleet-task-gen-run.sh
#
# Required env vars: WANDB_API_KEY, MODALITY, INFERENCE_BACKEND, LOGGER,
#   MAX_TURNS, MAX_INPUT_LENGTH, MAX_GENERATE_LENGTH, NUM_EPOCHS,
#   JUDGE_MODEL, K_ROLLOUTS, ALPHA, MAX_EVAL_STEPS
# SkyPilot env vars: SKYPILOT_NUM_GPUS_PER_NODE, SKYPILOT_NODE_IPS
set -euo pipefail

# Export RUN_NAME so task_gen_env can tag rollout dumps
# Always use random hex suffix for unique run names
export RUN_NAME="task_gen_$(python3 -c 'import os; print(os.urandom(4).hex())')"

# Optional: per-env dataset filtering via TASK_GEN_ENV_CLASSES env var
# e.g. TASK_GEN_ENV_CLASSES="outlook" or TASK_GEN_ENV_CLASSES="outlook,booking"
ENV_FILTER_ARGS=()
if [ -n "${TASK_GEN_ENV_CLASSES:-}" ]; then
  echo "=== env_filter: $TASK_GEN_ENV_CLASSES ==="
  ENV_FILTER_ARGS+=("data.env_filter=$TASK_GEN_ENV_CLASSES")
fi

# Task-gen GRPO training via shared run script
# --entrypoint: task-gen entrypoint (not main_fleet)
# --env-class: task_gen environment (not fleet_task)
# --data-dir-name: parquet files are in data/fleet/task_gen/ (not data/fleet/tool_use/)
# TP=1: N engines × 1 GPU each (Qwen3.5-9B fits in single H200)
# num_inference_engines auto-detected from SKYPILOT_NUM_GPUS_PER_NODE by fleet-common-run.sh
bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --entrypoint integrations.fleet.entrypoints.main_task_gen \
  --env-class task_gen -- \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3.5-9B" \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=${NUM_EPOCHS} \
  trainer.eval_batch_size=12 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=12 \
  trainer.use_hybrid_env_sampling=true \
  trainer.min_samples_per_env=1 \
  trainer.policy_mini_batch_size=12 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.loss_chunk_size=4096 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.sampling_params.temperature=0.95 \
  generator.sampling_params.top_p=0.95 \
  'generator.sampling_params.stop=["</tool_call>", "</task>"]' \
  generator.eval_sampling_params.temperature=0.95 \
  generator.eval_sampling_params.top_p=0.95 \
  'generator.eval_sampling_params.stop=["</tool_call>", "</task>"]' \
  trainer.policy.optimizer_config.lr=1.0e-6 \
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
  generator.gpu_memory_utilization=0.75 \
  trainer.logger="$LOGGER" \
  trainer.project_name="fleet-task-gen" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/task_gen" \
  trainer.dump_data_batch=true \
  ++environment.skyrl_gym.task_gen.max_turns=$MAX_TURNS \
  ++environment.skyrl_gym.task_gen.judge_model="$JUDGE_MODEL" \
  ++environment.skyrl_gym.task_gen.k_rollouts=$K_ROLLOUTS \
  ++environment.skyrl_gym.task_gen.alpha=$ALPHA \
  ++environment.skyrl_gym.task_gen.max_eval_steps=$MAX_EVAL_STEPS \
  ++environment.skyrl_gym.task_gen.evaluator_model="${EVALUATOR_MODEL:-anthropic/claude-sonnet-4.5}" \
  ++environment.skyrl_gym.task_gen.eval_k_rollouts=8 \
  ++environment.skyrl_gym.task_gen.tool_call_reward_per_call=0.02 \
  "${ENV_FILTER_ARGS[@]}" \
  "$@"
