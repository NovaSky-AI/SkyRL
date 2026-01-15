#!/bin/bash
set -x

# GOLD Integration Test Script for Modal
# This script runs a short GOLD training test on Modal GPUs.
#
# Usage (from skyrl-train/integrations/modal/):
#   MODAL_GPU=A100:2 modal run main.py --command "bash examples/gold_distillation/run_gold_test_modal.sh"
#
# This script tests cross-tokenizer distillation with minimal resources.

echo "=== GOLD Integration Test ==="
echo "Starting at: $(date)"

# Prepare GSM8K data
echo "=== Preparing GSM8K data ==="
uv run python examples/gold_distillation/prepare_gsm8k_data.py
DATA_DIR="/root/data/gsm8k"

# Cross-tokenizer test: Qwen (teacher) -> Llama (student)
# Using small models for quick testing
TEACHER_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
STUDENT_MODEL="meta-llama/Llama-3.2-1B"

# GOLD-specific settings
ADVANTAGE_ESTIMATOR="no_op"
POLICY_LOSS="importance_sampling"
USE_KL_IN_REWARD=true
USE_KL_LOSS=false

# Minimal settings for quick smoke test (~5 min)
NUM_GPUS_PER_NODE=2
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP_SIZE=1
TRAIN_BATCH_SIZE=8
MINI_BATCH_SIZE=8
N_SAMPLES_PER_PROMPT=2
EVAL_N_SAMPLES_PER_PROMPT=2
EPOCHS=1
LR=1e-5

# Output paths
CKPT_PATH="/root/data/gold_test_ckpt"
EXPORT_PATH="/root/data/gold_test_export"

echo "=== Starting GOLD Training Test ==="
echo "Teacher: $TEACHER_MODEL"
echo "Student: $STUDENT_MODEL"

# Use prepared GSM8K data
uv run --isolated --extra vllm -m examples.gold_distillation.main_gold_distill \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator=$ADVANTAGE_ESTIMATOR \
  trainer.algorithm.policy_loss_type=$POLICY_LOSS \
  trainer.policy.model.path=$STUDENT_MODEL \
  trainer.ref.model.path=$TEACHER_MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP_SIZE \
  trainer.epochs=$EPOCHS \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=false \
  trainer.eval_interval=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=1 \
  trainer.max_prompt_length=512 \
  generator.enforce_eager=true \
  generator.sampling_params.max_generate_length=256 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=0.7 \
  generator.eval_sampling_params.max_generate_length=256 \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.gpu_memory_utilization=0.7 \
  trainer.logger="none" \
  trainer.project_name="gold_test" \
  trainer.run_name="gold_integration_test" \
  trainer.resume_mode=none \
  trainer.export_path="$EXPORT_PATH" \
  trainer.hf_save_interval=1 \
  trainer.max_ckpts_to_keep=1 \
  trainer.ckpt_interval=1 \
  trainer.ckpt_path="$CKPT_PATH"

EXIT_CODE=$?

echo "=== GOLD Integration Test Complete ==="
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: GOLD training test passed!"
else
    echo "FAILURE: GOLD training test failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
