#!/usr/bin/env bash
# Compare adapter-only LoRA weight sync over disk vs. in memory.
#
# Megatron trainer (2 GPUs, EP=2) -> vLLM (1 engine, TP=2), non-colocated, on one
# 4xH100 node. Runs the same short GRPO job once per `lora.sync_mode` and prints
# the weight-sync timings and the rollout/trainer logprob gap for each, so the
# two modes can be compared for speed and agreement.
#
# GLM-4.7-Flash is a DeepSeek-style MoE (64 routed experts, top-4) whose adapter
# export is per-expert, so with share_expert_adapters=true the memory path's
# shared-expert aliasing is exercised. Override MODEL_NAME / MODES to change that.
#
# Usage: bash tests/train/gpu_e2e_test/lora_sync_mode_compare.sh
#   MODES="memory"  bash ...      # one mode only
set -euo pipefail

export CI=true

MODEL_NAME="${MODEL_NAME:-zai-org/GLM-4.7-Flash}"
MODES="${MODES:-disk memory}"
DATA_DIR="$HOME/data/gsm8k"
LOG_DIR="${LOG_DIR:-$HOME/lora_sync_compare}"
NUM_POLICY_GPUS=2
MEGATRON_EP=2
INFERENCE_TP=2
LORA_RANK=32
LORA_ALPHA=64
# 48 prompts / batch 16 = 3 optimizer steps, so 3 measured syncs after the initial one.
NUM_TRAIN_PROMPTS=48
TRAIN_BATCH_SIZE=16

mkdir -p "$LOG_DIR"
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir "$DATA_DIR" --max_train_dataset_length "$NUM_TRAIN_PROMPTS"

for MODE in $MODES; do
  RUN_NAME="lora_sync_${MODE}_$(date +%Y%m%d%H%M%S)"
  echo "=============================== sync_mode=$MODE ==============================="
  uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
    data.train_data="['$DATA_DIR/train.parquet']" \
    data.val_data="['$DATA_DIR/validation.parquet']" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.use_kl_loss=false \
    trainer.policy.model.path="$MODEL_NAME" \
    trainer.strategy=megatron \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_nodes=1 \
    trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
    generator.inference_engine.num_engines=1 \
    generator.inference_engine.tensor_parallel_size=$INFERENCE_TP \
    trainer.policy.megatron_config.tensor_model_parallel_size=1 \
    trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
    trainer.policy.megatron_config.context_parallel_size=1 \
    trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
    trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
    trainer.policy.model.lora.rank=$LORA_RANK \
    trainer.policy.model.lora.alpha=$LORA_ALPHA \
    trainer.policy.model.lora.sync_mode="$MODE" \
    trainer.policy.megatron_config.lora_config.merge_lora=false \
    trainer.epochs=1 \
    trainer.train_batch_size=$TRAIN_BATCH_SIZE \
    trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
    trainer.micro_forward_batch_size_per_gpu=2 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.update_epochs_per_batch=1 \
    generator.n_samples_per_prompt=4 \
    trainer.max_prompt_length=512 \
    generator.sampling_params.max_generate_length=512 \
    trainer.eval_before_train=false \
    trainer.eval_interval=0 \
    trainer.ckpt_interval=0 \
    trainer.hf_save_interval=0 \
    trainer.resume_mode=none \
    trainer.policy.optimizer_config.lr=1.0e-5 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.8 \
    generator.batched=true \
    environment.env_class=gsm8k \
    trainer.logger=console \
    trainer.project_name=lora_sync_compare \
    trainer.run_name="$RUN_NAME" \
    2>&1 | tee "$LOG_DIR/$MODE.log"
done

echo
echo "=============================== summary ==============================="
for MODE in $MODES; do
  echo "--- sync_mode=$MODE ---"
  # Per-sync timing and the rollout-vs-trainer logprob gap are driver-side, so they
  # are on stdout. The worker-side stage split ("LoRA sync ...") is not: SkyRL
  # redirects worker output to trainer.log_path/infra-<timestamp>.log via os.dup2,
  # so that file has to be read separately or the split silently comes back empty.
  grep -hE "timing/sync_weights|rollout_train_logprobs_abs_diff_mean" "$LOG_DIR/$MODE.log" 2>/dev/null \
    | sed -E 's/^.*(timing\/sync_weights[^,}]*|rollout_train_logprobs_abs_diff_mean[^,}]*).*$/\1/' \
    | tail -n 12 || true
  INFRA_LOG=$(grep -ohE "Infrastructure logs will be written to: \S+" "$LOG_DIR/$MODE.log" 2>/dev/null \
    | tail -n 1 | awk '{print $NF}')
  if [ -n "$INFRA_LOG" ] && [ -f "$INFRA_LOG" ]; then
    grep -hoE "LoRA sync[^|]*" "$INFRA_LOG" 2>/dev/null | sed -E 's/\x1b\[[0-9;]*m//g' | tail -n 6 || true
  else
    echo "  (worker infra log not found; per-stage split unavailable)"
  fi
done
