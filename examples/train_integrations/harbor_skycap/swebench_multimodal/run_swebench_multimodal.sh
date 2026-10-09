#!/usr/bin/env bash
# SWE-bench Multimodal through skycap: mini-swe-agent in Daytona sandboxes, a vision-language policy that
# sees the issue's screenshots, and every rollout's exact tokens and images captured for training.
#
#   SANDBOX_OWNER=<you> bash examples/train_integrations/harbor_skycap/swebench_multimodal/run_swebench_multimodal.sh
#
# Defaults: Qwen3.5-9B on 4 GPUs (FSDP, 4 vLLM engines), 8 tasks x 8 samples per step, 32k context.
# Prepares the tasks once (prepare_tasks.py: the dev split's pass-and-fail repos, screenshots copied
# under IMAGE_DIR), then trains EPOCHS passes over the chosen tasks.
#
#   TASKS=a,b                 task (instance) names; else the first NUM_PROMPTS
#   NUM_PROMPTS, GROUP_SIZE   tasks per step and samples per task (default 8 x 8)
#   EPOCHS, LR                passes over the tasks, learning rate (default 20, 1e-6). LR=0 with one epoch
#                             over many tasks measures each task's pass@GROUP_SIZE without training
#   MAX_CONCURRENCY           trials (so sandboxes) in flight at once, eval included (default 64). The
#                             Daytona org is shared: keep it well under what others leave free
#   SANDBOX_OWNER             required: the `owner` label on every sandbox; `run` is the experiment name.
#                             After a crash: python -m examples.train_integrations.harbor_skycap.daytona
#                             cleanup --label owner=$SANDBOX_OWNER --label run=<experiment>
#   SANDBOX_TTL_MINUTES       hard lifetime of a sandbox, whatever happens to this run (default 180)
#   SANDBOX_CPUS, SANDBOX_MEMORY_MB, SANDBOX_STORAGE_MB   per sandbox (default 1, 4096, 10240)
#   IMAGE_DIR                 the screenshots, at the same path on every node (skycap renders there)
#
# Needs: the GPUs, `DAYTONA_API_KEY=...` in DAYTONA_KEY_FILE, optionally WANDB_API_KEY (or WANDB_KEY_FILE).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"
HERE=examples/train_integrations/harbor_skycap/swebench_multimodal

#-----------------------
# Knobs
#-----------------------
: "${SANDBOX_OWNER:?set SANDBOX_OWNER, the owner label every sandbox of this run carries}"
DAYTONA_KEY_FILE="${DAYTONA_KEY_FILE:-$HOME/.config/skyrl/daytona.env}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/default/wandb_key.key}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TASKS="${TASKS:-}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1.0e-6}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
SANDBOX_TTL_MINUTES="${SANDBOX_TTL_MINUTES:-180}"
SANDBOX_CPUS="${SANDBOX_CPUS:-1}"
SANDBOX_MEMORY_MB="${SANDBOX_MEMORY_MB:-4096}"
SANDBOX_STORAGE_MB="${SANDBOX_STORAGE_MB:-10240}"
MINI_SWE_AGENT_VERSION="${MINI_SWE_AGENT_VERSION:-2.4.6}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-2400}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_ENGINES="${NUM_ENGINES:-4}"
TP_SIZE="${TP_SIZE:-1}"
SKYCAP_SERVERS="${SKYCAP_SERVERS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/swebench_multimodal}"
TASKS_DIR="$DATA_ROOT/tasks"
IMAGE_DIR="${IMAGE_DIR:-$DATA_ROOT/images}"

EXPERIMENT="swebm-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')"
RUN_DIR="${RUN_ROOT:-/tmp/harbor/runs}/$EXPERIMENT"
SUBSET_DIR="$RUN_DIR/tasks"

#-----------------------
# Credentials
#-----------------------
[[ -f "$DAYTONA_KEY_FILE" ]] || { echo "no Daytona credentials at $DAYTONA_KEY_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$DAYTONA_KEY_FILE"
set +a
: "${DAYTONA_API_KEY:?$DAYTONA_KEY_FILE must set DAYTONA_API_KEY}"
if [[ -z "${WANDB_API_KEY:-}" && -f "$WANDB_KEY_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$WANDB_KEY_FILE"
  set +a
fi
LOGGER="${LOGGER:-$([[ -n "${WANDB_API_KEY:-}" ]] && echo wandb || echo console)}"
[[ "$LOGGER" != wandb ]] || export WANDB_API_KEY

#-----------------------
# Data: the tasks once, then this run's subset
#-----------------------
if [[ -n "$(ls -A "$TASKS_DIR" 2>/dev/null)" ]]; then
  echo "==> tasks already at $TASKS_DIR"
else
  uv run --isolated --with "swebench==4.1.0" --with datasets --with pillow "$HERE/prepare_tasks.py" \
    --output-dir "$TASKS_DIR" --image-dir "$IMAGE_DIR"
fi
mkdir -p "$SUBSET_DIR"
if [[ -n "$TASKS" ]]; then
  IFS=, read -r -a tasks <<< "$TASKS"
  tasks=("${tasks[@]/#/$TASKS_DIR/}")
else
  mapfile -t tasks < <(find "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
  tasks=("${tasks[@]:0:$NUM_PROMPTS}")
fi
for task in "${tasks[@]}"; do
  [[ -d "$task" ]] || { echo "no task at $task" >&2; exit 1; }
  ln -s "$task" "$SUBSET_DIR/$(basename "$task")"
done
NUM_PROMPTS="${#tasks[@]}"
echo "==> $NUM_PROMPTS tasks x $GROUP_SIZE samples, at most $MAX_CONCURRENCY sandboxes at once"
echo "==> sandboxes labelled owner=$SANDBOX_OWNER run=$EXPERIMENT, ttl ${SANDBOX_TTL_MINUTES}m"

export RAY_ADDRESS=local
echo "==> experiment $EXPERIMENT in $RUN_DIR"

uv run --isolated --extra fsdp --extra harbor --extra skycap \
  -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap \
  data.train_data="['$SUBSET_DIR']" \
  trainer.policy.model.path="$MODEL" \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  trainer.project_name=skycap-swebench-multimodal \
  trainer.run_name="$EXPERIMENT" \
  trainer.logger="$LOGGER" \
  trainer.export_path="$RUN_DIR/exports" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=none \
  harbor_trial_config.trials_dir="$RUN_DIR/trials" \
  harbor_trial_config.agent.name=mini-swe-agent \
  harbor_trial_config.agent.override_timeout_sec="$AGENT_TIMEOUT_SEC" \
  harbor_trial_config.agent.kwargs.config_file="$REPO/$HERE/mini_swe_agent_textbased.yaml" \
  harbor_trial_config.agent.kwargs.version="$MINI_SWE_AGENT_VERSION" \
  harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.daytona:LabelledDaytonaEnvironment \
  harbor_trial_config.environment.kwargs.labels.owner="$SANDBOX_OWNER" \
  harbor_trial_config.environment.kwargs.labels.project=skyrl-swebm \
  harbor_trial_config.environment.kwargs.labels.run="$EXPERIMENT" \
  harbor_trial_config.environment.kwargs.ttl_minutes="$SANDBOX_TTL_MINUTES" \
  harbor_trial_config.environment.kwargs.auto_stop_interval_mins=30 \
  harbor_trial_config.environment.override_cpus="$SANDBOX_CPUS" \
  harbor_trial_config.environment.override_memory_mb="$SANDBOX_MEMORY_MB" \
  harbor_trial_config.environment.override_storage_mb="$SANDBOX_STORAGE_MB" \
  skycap.images=true \
  skycap.record_dir="$RUN_DIR/skycap" \
  skycap.num_servers="$SKYCAP_SERVERS" \
  skycap.exposure.type=cloudflare \
  skycap.train_paths=final \
  generator.merge_stepwise_output=false \
  generator.step_wise_trajectories=true \
  generator.apply_overlong_filtering=true \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.max_concurrency="$MAX_CONCURRENCY" \
  generator.n_samples_per_prompt="$GROUP_SIZE" \
  generator.sampling_params.temperature=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  trainer.epochs="$EPOCHS" \
  trainer.train_batch_size="$NUM_PROMPTS" \
  trainer.policy_mini_batch_size="$NUM_PROMPTS" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.remove_microbatch_padding=false \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.policy.optimizer_config.lr="$LR" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  "$@" 2>&1 | tee "$RUN_DIR/run.log"

echo
echo "==> done: $EXPERIMENT"
echo "    skycap records: $RUN_DIR/skycap"
echo "    Harbor trials:  $RUN_DIR/trials"
echo "    leftover sandboxes: python -m examples.train_integrations.harbor_skycap.daytona list \\"
echo "        --label owner=$SANDBOX_OWNER --label run=$EXPERIMENT"
