#!/usr/bin/env bash
# Eval-only run on Fleet envs with optional S3 checkpoint resume.
#
# When RESUME_RUN_NAME is set, downloads the latest FSDP checkpoint from S3,
# broadcasts it to worker nodes, loads policy weights, and runs a single eval
# pass. Eval results are dumped locally and uploaded to S3.
#
# When RESUME_RUN_NAME is unset, evaluates the base model at trainer.policy.model.path.
#
# Required env vars: FLEET_API_KEY, WANDB_API_KEY
# Optional env vars:
#   RESUME_RUN_NAME      Run name to resume from (S3 prefix). Empty = base model eval.
#   RESUME_CKPT_PATH     Local checkpoint dir to download into. Default: $HOME/ckpts/eval_only
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  (required for S3 resume / upload)
#   MODEL_PATH           HF model repo or path. Default: Qwen/Qwen3.5-9B
#   PROJECT_NAME         W&B / S3 project prefix. Default: fleet-tool-use-grpo
#   RUN_NAME             W&B run name + S3 eval upload prefix. Default: fleet_eval_only_<modality>_<n>
#   EVAL_N_SAMPLES       pass@K samples per prompt. Default: 8
#
set -euo pipefail
cd "$(dirname "$0")/.."  # cd to SkyRL root

export LOGGER="${LOGGER:-wandb}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
export MODALITY="${MODALITY:-tool_use}"
export MAX_TURNS="${MAX_TURNS:-50}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-72000}"
export MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-4096}"
export NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-8}"
export EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-8}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export S3_DATASET_BUCKET="${S3_DATASET_BUCKET:-fleet-internal-datasets}"
export S3_CHECKPOINT_BUCKET="${S3_CHECKPOINT_BUCKET:-skyrl-checkpoints}"
export S3_TRAJECTORY_BUCKET="${S3_TRAJECTORY_BUCKET:-skyrl-trajectories}"

: "${FLEET_API_KEY:?Set FLEET_API_KEY before running}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
PROJECT_NAME="${PROJECT_NAME:-fleet-tool-use-grpo}"
RUN_NAME="${RUN_NAME:-fleet_eval_only_${MODALITY}_pass_at_${EVAL_N_SAMPLES}}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-$HOME/ckpts/eval_only}"

# resume_mode controls how main_eval picks the checkpoint inside RESUME_CKPT_PATH.
# latest = read latest_ckpt_global_step.txt (written by S3 download); none = base weights.
if [ -n "${RESUME_RUN_NAME:-}" ]; then
  RESUME_MODE="${RESUME_MODE:-latest}"
else
  RESUME_MODE="${RESUME_MODE:-none}"
fi
export RESUME_RUN_NAME="${RESUME_RUN_NAME:-}"

DATA_ROOT=""
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
  DATA_ROOT="/workspace"
else
  DATA_ROOT="$HOME"
fi

EVAL_PARQUET="${DATA_ROOT}/data/fleet/${MODALITY}/validation.parquet"
TASKS_FILE="${DATA_ROOT}/data/fleet/tasks_${MODALITY}.json"

echo "=== Fleet Eval-Only Run ==="
echo "Model:           $MODEL_PATH"
echo "Project / Run:   $PROJECT_NAME / $RUN_NAME"
echo "Resume run name: ${RESUME_RUN_NAME:-(none — base model eval)}"
echo "Resume mode:     $RESUME_MODE"
echo "Ckpt path:       $RESUME_CKPT_PATH"
echo "Eval data:       $EVAL_PARQUET"
echo "Samples/prompt:  $EVAL_N_SAMPLES"

bash scripts/fleet-common-run.sh \
  --use-python-direct --cuda-env "$HOME/.cuda_env" \
  --set-ulimit --no-pytorch-alloc-conf \
  --entrypoint integrations.fleet.entrypoints.main_eval \
  --nccl-heartbeat 1800 -- \
  environment.skyrl_gym.fleet_task.ttl_seconds=900 \
  environment.skyrl_gym.fleet_task.partial_reward=true \
  environment.skyrl_gym.fleet_task.enable_hints=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.ckpt_path="$RESUME_CKPT_PATH" \
  trainer.eval_batch_size=4 \
  trainer.eval_interval=1 \
  trainer.max_prompt_length=2048 \
  trainer.dump_eval_results=true \
  trainer.export_path="$HOME/exports" \
  generator.chat_template_kwargs='{enable_thinking:true}' \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.eval_sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.eval_sampling_params.temperature=0.9 \
  generator.eval_sampling_params.top_p=0.95 \
  'generator.eval_sampling_params.stop=["</tool_call>"]' \
  generator.max_turns=$MAX_TURNS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES \
  generator.enforce_eager=false \
  generator.gpu_memory_utilization=0.65 \
  generator.inject_context_status=true \
  generator.context_warning_threshold=0.90 \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  "data.val_data=['${EVAL_PARQUET}']" \
  "environment.skyrl_gym.fleet_task.tasks_file=${TASKS_FILE}" \
  "$@"
