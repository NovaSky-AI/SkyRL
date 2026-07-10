set -ex

# Companion to run_codecontest.sh — same Harbor recipe, routed through the
# Arctic RL backend by adding one flag to Harbor's own entrypoint:
#     trainer.override_entrypoint=integrations.arctic_rl.harbor_entrypoint
# Everything else that changes below is mechanical fallout of routing through
# the Arctic (DeepSpeed + vLLM) stack instead of FSDP + SkyRL's own inference
# engines: different uv extras, disaggregated placement, and a handful of
# `trainer.arctic_rl.*` topology knobs. Full design + tier caveats:
# integrations/arctic_rl/docs/HARBOR_DESIGN.md.

# wandb api key.
# export WANDB_API_KEY=YOUR_KEY_HERE

# Pick the sandbox provider and provide the credentials.
# export DAYTONA_API_KEY=YOUR_KEY_HERE
# export MODAL_TOKEN_ID=YOUR_KEY_HERE
# export MODAL_TOKEN_SECRET=YOUR_KEY_HERE

: "${WANDB_API_KEY:?WANDB_API_KEY not set}"
: "${DAYTONA_API_KEY:?DAYTONA_API_KEY not set (or pick another sandbox provider)}"

# Bypass any global gitconfig url.insteadOf HTTPS->SSH rewrites — the target
# training env firewalls SSH, and uv needs plain HTTPS to clone
# arctic-platform / arctic-inference / megatron-core.
export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-/dev/null}"

# The Arctic OpenAI shim (integrations/arctic_rl/openai_bridge.py) binds here
# and Harbor's LiteLLM client calls it. Upstream removed SkyRL's HTTP-endpoint
# config keys, so host/port flow through env vars to avoid new config surface.
export ARCTIC_HARBOR_SHIM_HOST="${ARCTIC_HARBOR_SHIM_HOST:-0.0.0.0}"
export ARCTIC_HARBOR_SHIM_PORT="${ARCTIC_HARBOR_SHIM_PORT:-8000}"

# integrations.arctic_rl.* imports must resolve inside the Ray driver task
# (uv+ray plugin replays this same invocation on workers, but PYTHONPATH from
# the launcher makes the first import succeed before the plugin kicks in).
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
# All knobs below accept env-var overrides so you can do e.g.
#     NUM_POLICY_GPUS=4 MODEL=Qwen/Qwen3-0.6B MAX_MODEL_LEN=8192 \
#         bash examples/train_integrations/harbor/run_codecontest_arctic.sh
# for a fast smoke without editing the file.
: "${MODEL:=Qwen/Qwen3-8B}"
: "${SERVED_MODEL_NAME:=$(basename "${MODEL}")}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${MINI_BATCH_SIZE:=32}"
: "${MAX_MODEL_LEN:=32768}"

# Algorithmic parameters
LOSS_REDUCTION="token_mean"  # with step-wise training, we have to use token_mean to be prefix-merge-invariant
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false
APPLY_OVERLONG_FILTERING=true

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
: "${NUM_INFERENCE_ENGINES:=${NUM_POLICY_GPUS}}"   # arctic disaggregates: one vLLM replica per GPU
: "${TP_SIZE:=1}"
: "${ENABLE_RATE_LIMITING:=true}"        # Enable rate/concurrency limiting for trajectory submissions
: "${TRAJECTORIES_PER_SECOND:=5}"        # Maximum trajectories per second (must be >= 1.0, fractional values like 1.5 are supported). null or omit to disable rate limiting
: "${MAX_CONCURRENCY:=512}"              # Maximum concurrent trial.run() calls allowed (must be >= 1). null or omit to disable concurrency limiting

#----------------
# Arctic RL knobs (defaults match integrations/arctic_rl/examples/run_codecontest_arctic_harbor.sh)
#----------------
COLOCATE=true                    # share GPUs between DeepSpeed trainer + vLLM inference
ZERO_STAGE=3                     # 8B under colocation needs ZeRO-3
USE_LIGER=true
USE_ARCTIC_INFERENCE=true        # Forest Cascade Attention on the rollout
USE_ZORRO=false                  # prompt-group dedup; leave off for Harbor's non-prefix-shared prompts
CUDA_IPC_WEIGHT_SYNC=true        # near-zero-copy weight transfer in colocated mode
LOW_MEM_WEIGHT_SYNC=false
VLLM_ENFORCE_EAGER=false         # graph-compile pays off after step 1
VLLM_GPU_MEM_UTIL=0.7
ATTN_IMPL=flash_attention_2      # FA3 requires the FA3 wheel; FA2 is safe on Hopper

# arctic-inference pins vLLM 0.18 + torch 2.10; the default flash-attn wheel
# in SkyRL's lock targets torch 2.11, so we override with a matching wheel.
FLASH_ATTN_WHL="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# uv env swap notes (why this differs from the FSDP launcher):
#   * `--isolated` bypasses the project lock: arctic-inference[vllm] needs
#     vLLM 0.18 + torch 2.10, which conflict with the fsdp/megatron pins.
#   * `--extra skyrl-train --extra harbor` pulls the base training deps +
#     harbor[daytona,modal]. No `--extra fsdp`: Arctic ships its own trainer.
#   * `--with arctic-platform arctic-inference[vllm] liger-kernel` provides
#     the arctic stack.
#   * `--with 'transformers==4.57.6'` is exact-pinned (not `<5`): raylet
#     re-spawns workers via `bash -c`, and the shell would parse the
#     unquoted `<5` as a redirect from fd 5 and kill the worker.
#   * `--with "flash-attn@..."` overrides SkyRL's default torch-2.11 flash-
#     attn wheel to match arctic-inference's torch 2.10 pin.
# SkyRL's uv+ray plugin replays this exact invocation on every Ray worker via
# `py_executable`, so workers get the same env automatically.

# Run SkyRL command
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
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens=$MAX_MODEL_LEN \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens=$MAX_MODEL_LEN \
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
