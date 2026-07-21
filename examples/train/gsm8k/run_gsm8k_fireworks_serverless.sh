#!/usr/bin/env bash

set -euo pipefail
set -x

# One-step, policy-only GRPO smoke test on Fireworks serverless training.
# qwen3-1p7b is a low-cost candidate, but serverless model enablement is
# account-specific during private preview. Override both model variables if it
# is not enabled for your account.

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${FIREWORKS_BASE_MODEL:=accounts/fireworks/models/qwen3-1p7b}"
: "${FIREWORKS_TOKENIZER_MODEL:=Qwen/Qwen3-1.7B}"
: "${FIREWORKS_MAX_SEQ_LEN:=32768}"
: "${FIREWORKS_LORA_RANK:=8}"
: "${TRAIN_BATCH_SIZE:=2}"
: "${N_SAMPLES_PER_PROMPT:=4}"
: "${MAX_GENERATE_LENGTH:=256}"
: "${LOGGER:=wandb}"
: "${RUN_NAME:=gsm8k-fireworks-${FIREWORKS_BASE_MODEL##*/}-serverless-grpo-smoke}"

if [[ "${FIREWORKS_RUN_CONFIRMED:-0}" != "1" ]]; then
  set +x
  printf '%s\n' \
    "This command opens a real paid Fireworks serverless training session." \
    "Resolved smoke-test plan:" \
    "  base model: ${FIREWORKS_BASE_MODEL}" \
    "  tokenizer: ${FIREWORKS_TOKENIZER_MODEL}" \
    "  run name: ${RUN_NAME}" \
    "  LoRA rank: ${FIREWORKS_LORA_RANK}" \
    "  max sequence length: ${FIREWORKS_MAX_SEQ_LEN}" \
    "  optimizer steps: 1" \
    "  prompt groups: ${TRAIN_BATCH_SIZE}" \
    "  completions per prompt: ${N_SAMPLES_PER_PROMPT}" \
    "  maximum generated tokens per completion: ${MAX_GENERATE_LENGTH}" \
    "  provider operations: 1 session, 1 forward/backward, 1 optimizer step" \
    "  sampler snapshots: 2 (initial weights and post-step weights)" \
    "  maximum sampled tokens: $(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT * MAX_GENERATE_LENGTH ))" \
    "  checkpoint/export: disabled" \
    "Verify that the model is enabled for serverless training on your account," \
    "then rerun with FIREWORKS_RUN_CONFIRMED=1."
  exit 2
fi

# Keep credential values out of shell tracing. The SDK reads them directly
# from the environment; they are never passed as CLI/config values.
set +x
if [[ -z "${FIREWORKS_API_KEY:-}" ]]; then
  printf 'FIREWORKS_API_KEY is not set in this shell.\n' >&2
  exit 2
fi
if [[ "$LOGGER" == "wandb" && -z "${WANDB_API_KEY:-}" ]]; then
  printf 'WANDB_API_KEY is not set in this shell (or set LOGGER=console).\n' >&2
  exit 2
fi
set -x

uv run --isolated --extra fireworks -m skyrl.train.entrypoints.main_fireworks \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="[]" \
  trainer.strategy=fireworks \
  trainer.fireworks.infrastructure=serverless \
  trainer.fireworks.base_model="$FIREWORKS_BASE_MODEL" \
  trainer.fireworks.max_seq_len="$FIREWORKS_MAX_SEQ_LEN" \
  trainer.fireworks.snapshot_prefix=gsm8k-fireworks-smoke \
  trainer.policy.model.path="$FIREWORKS_TOKENIZER_MODEL" \
  trainer.policy.model.lora.rank="$FIREWORKS_LORA_RANK" \
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.policy.optimizer_config.adam_betas="[0.9,0.95]" \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.scheduler=constant_with_warmup \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=rollout_is \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.zero_variance_filter=true \
  trainer.critic.model.path=null \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=false \
  trainer.epochs=1 \
  trainer.max_training_steps=1 \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))" \
  trainer.micro_train_batch_size_per_gpu="$(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))" \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=512 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=null \
  trainer.enable_ray_gpu_monitor=false \
  generator.inference_engine.backend=fireworks \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.enable_ray_prometheus_stats=false \
  generator.batched=false \
  generator.max_turns=1 \
  generator.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT" \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.logprobs=1 \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name=gsm8k-fireworks \
  trainer.run_name="$RUN_NAME" \
  "$@"
