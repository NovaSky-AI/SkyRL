#!/usr/bin/env bash
set -x

# OpenReward + SkyRL training on Modal (4x GPU).
#
# Prerequisites:
#   1. export OPENREWARD_API_KEY="your-key"
#   2. Prepare dataset:
#      python examples/train_integrations/openreward/prepare_tasks.py \
#        --env "GeneralReasoning/WhoDunit" --split train --output /root/data/openreward/train.parquet
#
# Usage (via Modal):
#   MODAL_GPU=A100:4 modal run examples/train_integrations/modal/main.py \
#     --command "OPENREWARD_API_KEY=... WANDB_API_KEY=... bash examples/train_integrations/openreward/run_openreward.sh"
#
# Override any config via positional args:
#   bash run_openreward.sh trainer.epochs=2

DATA_DIR="${DATA_DIR:-/root/data/openreward}"
CKPT_DIR="${CKPT_DIR:-/root/data/ckpts/openreward}"
EXPORT_DIR="${EXPORT_DIR:-/root/data/export/openreward}"
RUN_NAME="${RUN_NAME:-skyrl-openreward-whodunit}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

: "${NUM_GPUS:=4}"
: "${LOGGER:=wandb}"

uv run --isolated --extra fsdp --with openreward \
  -m examples.train_integrations.openreward.entrypoints.main_openreward \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/train.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  generator.inference_engine.async_engine=true \
  trainer.epochs=3 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=1024 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.n_samples_per_prompt=4 \
  generator.max_turns=10 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</tool_call>"]' \
  environment.env_class="openreward" \
  environment.skyrl_gym.max_env_workers=16 \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyrl-openreward" \
  trainer.run_name="${RUN_NAME}" \
  trainer.ckpt_interval=20 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</tool_call>"]' \
  generator.eval_sampling_params.max_generate_length=1024 \
  trainer.export_path="$EXPORT_DIR" \
  trainer.eval_interval=50 \
  $@
