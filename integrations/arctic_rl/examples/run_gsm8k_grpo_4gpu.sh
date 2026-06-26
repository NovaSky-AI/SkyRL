#!/usr/bin/env bash
# GSM8K GRPO training via Arctic RL server.
#
# Invoked from the SkyRL repo root via core dispatch:
#   python -m skyrl.train.entrypoints.main_base \
#       trainer.override_entrypoint=integrations.arctic_rl.entrypoint <flags>
#
# Non-colocated (4 GPUs: 2 training + 2 sampling):
#   bash integrations/arctic_rl/examples/run_gsm8k_grpo_4gpu.sh \
#     trainer.placement.policy_num_gpus_per_node=2 \
#     generator.inference_engine.num_engines=2 \
#     trainer.policy_mini_batch_size=4
#
# Colocated (4 GPUs: all shared between training + sampling):
#   bash integrations/arctic_rl/examples/run_gsm8k_grpo_4gpu.sh \
#     trainer.arctic_rl.colocate=true \
#     trainer.placement.policy_num_gpus_per_node=4 \
#     generator.inference_engine.num_engines=4 \
#     generator.inference_engine.gpu_memory_utilization=0.3 \
#     trainer.policy_mini_batch_size=8

set -euo pipefail

export PYTHONUNBUFFERED=1

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${MODEL:="Qwen/Qwen3-0.6B"}"
: "${LOGGER:="console"}"

# Auto-prep GSM8K parquets if missing (uses SkyRL's bundled prep script).
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/validation.parquet" ]]; then
    REPO_ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
    echo "GSM8K parquets not found in ${DATA_DIR} — running SkyRL prep script..."
    python "${REPO_ROOT}/examples/train/gsm8k/gsm8k_dataset.py" --output_dir "${DATA_DIR}"
fi

python -m skyrl.train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="${MODEL}" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=2 \
  trainer.epochs=1 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=256 \
  trainer.policy_mini_batch_size=4 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.7e-6 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.logger="${LOGGER}" \
  trainer.project_name=arctic_rl \
  trainer.run_name="skyrl_gsm8k_${MODEL##*/}" \
  trainer.resume_mode=null \
  trainer.log_path=/tmp/skyrl-arctic-logs \
  trainer.ckpt_path="${HOME}/ckpts/skyrl_gsm8k_arctic" \
  "$@"
