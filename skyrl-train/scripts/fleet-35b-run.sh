#!/usr/bin/env bash
# Single source of truth for Qwen3.5-35B-A3B GRPO training config.
# Called by the SkyPilot YAML and by fleet-research run.sh.
# Expects env vars (FLEET_API_KEY, WANDB_API_KEY, AWS creds, etc.) exported before calling.
set -euo pipefail
cd "$(dirname "$0")/../.."  # cd to SkyRL root

bash skyrl-train/scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --nccl-heartbeat 1800 -- \
  environment.skyrl_gym.fleet_task.ttl_seconds=900 \
  environment.skyrl_gym.fleet_task.partial_reward=true \
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
  trainer.project_name="fleet-task-grpo" \
  trainer.run_name="fleet_qwen35_35b_${MODALITY}_${RUN_ID:-$(head -c 4 /dev/urandom | xxd -p)}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/fleet_qwen35_35b_${MODALITY}" \
  trainer.export_path="$HOME/exports" \
  trainer.dump_data_batch=true \
  "$@"
