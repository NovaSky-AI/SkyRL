set -ex

# run_codecontest.sh with the Arctic RL backend spliced in. Diff versus the
# FSDP baseline: swap uv extras, add `trainer.override_entrypoint=...`, add
# `trainer.arctic_rl.*`. See integrations/arctic_rl/docs/HARBOR_DESIGN.md.

# export WANDB_API_KEY=...
# export DAYTONA_API_KEY=...        # or MODAL_TOKEN_ID/_SECRET, E2B_API_KEY
: "${WANDB_API_KEY:?WANDB_API_KEY not set}"
: "${DAYTONA_API_KEY:?DAYTONA_API_KEY not set (or pick another sandbox provider)}"

# Some envs firewall SSH; force plain HTTPS for uv git clones.
export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-/dev/null}"

# Bind + advertise addresses for the Arctic OpenAI shim (openai_bridge.py).
export ARCTIC_HARBOR_SHIM_HOST="${ARCTIC_HARBOR_SHIM_HOST:-0.0.0.0}"
export ARCTIC_HARBOR_SHIM_PORT="${ARCTIC_HARBOR_SHIM_PORT:-8000}"

# Repo root on PYTHONPATH so the driver Ray task can import integrations.arctic_rl.*.
_REPO_ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
export PYTHONPATH="${_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

#-----------------------
# Dataset setup
#-----------------------
# Prepare datasets first (downloads from HuggingFace and extracts tasks):
# uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests
# uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/OpenThoughts-TB-dev
DATA_DIR="$HOME/data/harbor"
TRAIN_DATA="['$DATA_DIR/CodeContests']"
EVAL_DATA="['$DATA_DIR/OpenThoughts-TB-dev']"

#-----------------------
# Directory setup
#-----------------------
RUN_NAME="codecontest_arctic"
STORAGE_ROOT="/mnt/local_storage/$RUN_NAME"
TRIALS_DIR="$STORAGE_ROOT/trials_run"
CKPTS_DIR="$STORAGE_ROOT/ckpts"
EXPORTS_DIR="$STORAGE_ROOT/exports"
LOG_DIR="$STORAGE_ROOT/logs"

#-----------------------
# Training setup
#-----------------------
# Env-overridable knobs. Smoke on a smaller model with e.g.
#   NUM_POLICY_GPUS=4 MODEL=Qwen/Qwen3-0.6B MAX_MODEL_LEN=8192 bash $0
: "${MODEL:=Qwen/Qwen3-8B}"
: "${SERVED_MODEL_NAME:=$(basename "${MODEL}")}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${MINI_BATCH_SIZE:=32}"
: "${MAX_MODEL_LEN:=32768}"

# Algorithmic parameters
LOSS_REDUCTION="token_mean"  # with step-wise training, we have to use token_mean to be prefix-merge-invariant
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false
# Keep overlong trajectories in the batch under Arctic. Upstream
# arctic_platform.rl.utils.batch.split_dict raises IndexError if the surviving
# batch shrinks below num_workers, so we can't afford to drop samples. Overlong
# trajectories still contribute a zero-reward learning signal.
APPLY_OVERLONG_FILTERING=false

# Essentially achieves interleaved thinking (does not strip thinking tokens). Allows our step-wise
# training to be able to merge more step-wise outputs and hence speed up training.
# If you change the model you train, please change it accordingly, and decide if you need to make
# modifications.
CHAT_TEMPLATE_PATH="$(dirname "$0")/../../../skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"

# TIS corrections
TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

#----------------
# Infrastructure setup
#----------------
: "${NUM_POLICY_GPUS:=8}"
: "${NUM_INFERENCE_ENGINES:=${NUM_POLICY_GPUS}}"  # arctic runs one vLLM replica per GPU
: "${TP_SIZE:=1}"
: "${ENABLE_RATE_LIMITING:=true}"
: "${TRAJECTORIES_PER_SECOND:=5}"
: "${MAX_CONCURRENCY:=512}"

#----------------
# Arctic RL knobs — see integrations/arctic_rl/README.md for the full table.
#----------------
COLOCATE=true
ZERO_STAGE=3
USE_LIGER=true
USE_ARCTIC_INFERENCE=true
USE_ZORRO=false                  # off for Harbor: prompts don't share the (problem x N samples) prefix ZoRRo dedupes on
CUDA_IPC_WEIGHT_SYNC=true
LOW_MEM_WEIGHT_SYNC=false
VLLM_ENFORCE_EAGER=false
VLLM_GPU_MEM_UTIL=0.7
ATTN_IMPL=flash_attention_2

# torch-2.10 flash-attn wheel matching arctic-inference's vLLM 0.18 pin
# (SkyRL's default lock ships a torch-2.11 wheel).
FLASH_ATTN_WHL="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# Preflight: surface leaked ports / busy GPUs from a previous crashed run so we
# don't fail 5 minutes into engine init with a cryptic EADDRINUSE.
if command -v ss >/dev/null 2>&1 && ss -Hltn "sport = :${ARCTIC_HARBOR_SHIM_PORT}" 2>/dev/null | grep -q .; then
    echo "!! Port ${ARCTIC_HARBOR_SHIM_PORT} (Arctic OpenAI shim) is already in use." >&2
    echo "   Rerun with ARCTIC_HARBOR_SHIM_PORT=<free port>, or clean stale workers:" >&2
    echo "     pkill -9 -f 'InferenceWorker|DeepSpeedWorker|VLLM::|ArcticRLRayServerState|skyrl_entrypoint' && sleep 5" >&2
    exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 1024' | wc -l)
    if [[ "${busy}" -gt 0 ]]; then
        echo "!! WARNING: ${busy} GPU(s) already have >1 GiB memory used." >&2
        echo "   If a previous Arctic run crashed, free them with:" >&2
        echo "     pkill -9 -f 'InferenceWorker|DeepSpeedWorker|VLLM::|ArcticRLRayServerState|skyrl_entrypoint' && sleep 5" >&2
    fi
fi

uv run --isolated --extra skyrl-train --extra harbor \
  --with arctic-platform \
  --with 'arctic-inference[vllm]' \
  --with liger-kernel \
  --with 'transformers==4.57.6' \
  --with "flash-attn@${FLASH_ATTN_WHL}" \
  -m examples.train_integrations.harbor.entrypoints.main_harbor \
  trainer.override_entrypoint=integrations.arctic_rl.harbor_entrypoint \
  data.train_data=$TRAIN_DATA \
  data.val_data=$EVAL_DATA \
  trainer.policy.model.path=$MODEL \
  generator.inference_engine.served_model_name=$SERVED_MODEL_NAME \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.arctic_rl.colocate=$COLOCATE \
  trainer.arctic_rl.zero_stage=$ZERO_STAGE \
  trainer.arctic_rl.use_liger=$USE_LIGER \
  trainer.arctic_rl.use_arctic_inference=$USE_ARCTIC_INFERENCE \
  trainer.arctic_rl.use_zorro=$USE_ZORRO \
  trainer.arctic_rl.cuda_ipc_weight_sync=$CUDA_IPC_WEIGHT_SYNC \
  trainer.arctic_rl.low_memory_weight_sync=$LOW_MEM_WEIGHT_SYNC \
  trainer.arctic_rl.attn_implementation=$ATTN_IMPL \
  trainer.arctic_rl.vllm_enforce_eager=$VLLM_ENFORCE_EAGER \
  trainer.arctic_rl.vllm_max_model_len=$MAX_MODEL_LEN \
  trainer.arctic_rl.vllm_enable_prefix_caching=true \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.gpu_memory_utilization=$VLLM_GPU_MEM_UTIL \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  trainer.epochs=3 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$MINI_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.max_ckpts_to_keep=5 \
  trainer.hf_save_interval=5 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=true \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=2 \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  trainer.logger=wandb \
  trainer.project_name=harbor \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  generator.inference_engine.enforce_eager=$VLLM_ENFORCE_EAGER \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  "$@"
