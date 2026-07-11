#!/usr/bin/env bash
# Arctic RL + Harbor CodeContests GRPO smoke.
#
# Dispatch mechanism: this script invokes Harbor's existing entrypoint
# (``examples.train_integrations.harbor.entrypoints.main_harbor``) with one
# extra CLI flag —
#   trainer.override_entrypoint=integrations.arctic_rl.harbor_entrypoint
# — mirroring the way any SkyRL recipe toggles to Arctic RL. All the
# Harbor knobs below are the same knobs the FSDP baseline
# (``examples/train_integrations/harbor/run_codecontest.sh``) accepts;
# switching backend is a one-line change in your own launcher.

set -euo pipefail
export PYTHONUNBUFFERED=1

# ---------- credentials ----------
: "${WANDB_API_KEY:?WANDB_API_KEY not set; export it or source env.sh}"
: "${DAYTONA_API_KEY:?DAYTONA_API_KEY not set (or pick another sandbox provider)}"
: "${WANDB_BASE_URL:=https://snowflake.wandb.io}"
: "${WANDB_PROJECT:=arctic_rl_harbor}"
export WANDB_API_KEY WANDB_BASE_URL WANDB_PROJECT DAYTONA_API_KEY
# Bypass user's global gitconfig url.insteadOf that rewrites github HTTPS -> SSH
# (SSH port is firewalled in the target env). Required for the uv resolver to
# clone arctic-platform / arctic-inference / harbor / megatron-core.
export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-/dev/null}"
# Reuse a persistent HF cache across runs so we don't re-download Qwen weights
# on every smoke.
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# SkyRL's driver task deserializes ``integrations.arctic_rl.*`` +
# ``examples.train_integrations.harbor.*`` inside a fresh Ray worker; add the
# repo root to PYTHONPATH so those imports resolve without relying on the
# entrypoint's own env-forwarding fallback (which only fires on the driver
# task, not on the transient uv resolver process that spawns it).
_REPO_ROOT="$(cd "$(dirname "$0")"/../../.. && pwd)"
export PYTHONPATH="${_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---------- topology ----------
: "${NUM_GPUS:=4}"
: "${MODEL:=Qwen/Qwen3-0.6B}"
: "${SERVED_MODEL_NAME:=$(basename "${MODEL}")}"
# Match the Harbor FSDP baseline recipe (examples/train_integrations/harbor/
# run_codecontest.sh): 32K vLLM context, with 24K reserved for the packed
# multi-turn prompt and 8K for the response so Harbor's agent never
# overshoots the vLLM engine's max_model_len mid-trial.
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_PROMPT_LENGTH:=24576}"
: "${MAX_GENERATE_LENGTH:=8192}"

# ---------- run + data layout ----------
RUN_NAME="${RUN_NAME:-arctic_harbor_codecontest_$(basename "${MODEL}")_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/data/skyrl-runs/arctic_harbor}"
# Persistent, deduplicated Harbor task cache (prepare_harbor_dataset.py extracts
# each dataset once into a subdir). Falls back to $HOME/data/harbor for parity
# with the upstream FSDP launcher.
: "${DATA_DIR:=${RUN_ROOT}/data}"
: "${TRAIN_DATA:=[\"${DATA_DIR}/CodeContests\"]}"
: "${EVAL_DATA:=[\"${DATA_DIR}/CodeContests\"]}"   # reuse train for smoke; override for real eval

TRIALS_DIR="${TRIALS_DIR:-${RUN_ROOT}/trials/${RUN_NAME}}"
CKPTS_DIR="${CKPTS_DIR:-${RUN_ROOT}/ckpts/${RUN_NAME}}"
EXPORTS_DIR="${EXPORTS_DIR:-${RUN_ROOT}/exports/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs/${RUN_NAME}}"
mkdir -p "${TRIALS_DIR}" "${CKPTS_DIR}" "${EXPORTS_DIR}" "${LOG_DIR}"

# ---------- batch math ----------
# Small-by-default to keep the smoke under 10 min on Qwen3-0.6B. Scale up
# via `trainer.train_batch_size=... trainer.policy_mini_batch_size=...` on
# the CLI when moving off smoke.
: "${TRAIN_BATCH_SIZE:=8}"
: "${MINI_BATCH_SIZE:=8}"
: "${N_SAMPLES_PER_PROMPT:=4}"
: "${LR:=1.0e-6}"
: "${EPOCHS:=1}"
: "${EVAL_INTERVAL:=0}"          # 0 = never; smoke doesn't run Harbor eval
: "${EVAL_BEFORE_TRAIN:=false}"

# ---------- rate limiting (Harbor sandbox back-pressure) ----------
: "${ENABLE_RATE_LIMITING:=true}"
: "${TRAJECTORIES_PER_SECOND:=5}"
: "${MAX_CONCURRENCY:=64}"

# ---------- Arctic RL knobs ----------
# Defaults match the arctic BIRD-8B recipe (run_bird_grpo_8b_32gpu.sh) with
# Harbor-specific safety flips (ZoRRO off — untested through the OpenAI shim).
: "${COLOCATE:=true}"
: "${ZERO_STAGE:=3}"                    # 8B under colocation needs ZeRO-3 to fit vLLM alongside
: "${USE_LIGER:=true}"
: "${USE_ARCTIC_INFERENCE:=true}"       # FCA + Arctic spec-dec on server-side vLLM
: "${USE_ZORRO:=false}"                 # prompt-group dedup + chunked logits; enable once shape assertions verified
: "${CUDA_IPC_WEIGHT_SYNC:=true}"       # near-zero-copy weight transfer in colocated mode
: "${LOW_MEM_WEIGHT_SYNC:=false}"
: "${VLLM_ENFORCE_EAGER:=false}"        # graph-compile pays off after step 1
: "${VLLM_GPU_MEM_UTIL:=0.7}"
: "${ATTN_IMPL:=flash_attention_2}"     # FA3 requires the FA3 wheel; FA2 is safe on Hopper

# ---------- Dr. GRPO / loss knobs (parity with the FSDP baseline recipe) ----------
# Note: with step-wise Harbor training, upstream's run_codecontest.sh uses
# ``token_mean`` for prefix-merge invariance. If you flip this on your own
# launcher, keep it consistent with generator.merge_stepwise_output below.
: "${LOSS_REDUCTION:=token_mean}"
: "${GRPO_NORM_BY_STD:=false}"
: "${USE_KL_LOSS:=false}"
# Keep overlong trajectories in the batch under Arctic: upstream
# arctic_platform.rl.utils.batch.split_dict raises IndexError if the surviving
# batch shrinks below num_workers, so we can't afford to drop samples. Trajectories
# that overflow still get a zero reward, which is the right learning signal.
: "${APPLY_OVERLONG_FILTERING:=false}"
: "${MICRO_FWD_BATCH_PER_GPU:=1}"
: "${MICRO_TRAIN_BATCH_PER_GPU:=1}"

# ---------- Harbor HTTP endpoint (Arctic OpenAI shim serves here) ----------
# The shim binds on HTTP_ENDPOINT_HOST and starts port-searching from
# HTTP_ENDPOINT_PORT (upstream removed the corresponding SkyRL config keys;
# these now flow through env vars so we don't add new config surface).
: "${HTTP_ENDPOINT_HOST:=0.0.0.0}"
: "${HTTP_ENDPOINT_PORT:=8000}"
export ARCTIC_HARBOR_SHIM_HOST="${HTTP_ENDPOINT_HOST}"
export ARCTIC_HARBOR_SHIM_PORT="${HTTP_ENDPOINT_PORT}"

# Chat template — use SkyRL's bundled Qwen3 thinking template for parity with
# the FSDP Harbor baseline (see examples/train_integrations/harbor/run_codecontest.sh).
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-$(cd "$(dirname "$0")"/../../.. && pwd)/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2}"

if [[ ! -d "${DATA_DIR}/CodeContests" ]]; then
    echo "!! CodeContests tasks not found at ${DATA_DIR}/CodeContests." >&2
    echo "!! Run: uv run --extra harbor examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests --output_dir ${DATA_DIR}/CodeContests" >&2
    exit 1
fi

echo "==> Launching Arctic RL + Harbor CodeContests"
echo "    RUN_NAME=${RUN_NAME}"
echo "    MODEL=${MODEL} (served as '${SERVED_MODEL_NAME}')"
echo "    NUM_GPUS=${NUM_GPUS}  colocate=${COLOCATE}  zero_stage=${ZERO_STAGE}"
echo "    train_batch=${TRAIN_BATCH_SIZE} x ${N_SAMPLES_PER_PROMPT} samples/prompt"
echo "    HTTP endpoint: ${HTTP_ENDPOINT_HOST}:${HTTP_ENDPOINT_PORT}"
echo "    Trials dir: ${TRIALS_DIR}"
echo "    Log dir: ${LOG_DIR}"
echo

# Driver: same shape as the stock arctic_rl launchers
# (integrations/arctic_rl/examples/run_gsm8k_grpo_4gpu.sh) so no changes to
# pyproject.toml are needed.
#   - ``--isolated`` bypasses the project lock: arctic-inference[vllm] needs
#     vLLM 0.18 + torch 2.10, which conflict with the fsdp/megatron pins in
#     ``[tool.uv.sources]`` (vLLM 0.23 / torch 2.11).
#   - ``--extra skyrl-train`` pulls the base training deps (ray, deepspeed,
#     fastapi, wandb, ...); ``--extra harbor`` layers harbor[daytona,modal]
#     on top. Harbor doesn't pin vLLM/torch so it composes cleanly.
#   - ``--with arctic-platform arctic-inference[vllm] liger-kernel`` provides
#     the arctic stack.
#   - ``--with 'transformers==4.57.6'`` is exact-pinned (not ``<5``): raylet
#     re-spawns workers via ``bash -c``, and the shell would parse the
#     unquoted ``<5`` as a redirect from fd 5 and kill the worker.
#   - ``--with "flash-attn@<torch-2.10 wheel URL>"`` overrides SkyRL's default
#     torch-2.11 flash-attn wheel to match arctic-inference's torch 2.10 pin.
# SkyRL's uv+ray plugin replays this exact invocation on every Ray worker via
# ``py_executable``, so workers get the same env automatically — no
# ``uv pip install`` bootstrap needed.
FLASH_ATTN_WHL="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

exec uv run --isolated --extra skyrl-train --extra harbor \
    --with arctic-platform \
    --with 'arctic-inference[vllm]' \
    --with liger-kernel \
    --with 'transformers==4.57.6' \
    --with "flash-attn@${FLASH_ATTN_WHL}" \
    -m examples.train_integrations.harbor.entrypoints.main_harbor \
    trainer.override_entrypoint=integrations.arctic_rl.harbor_entrypoint \
    \
    data.train_data="${TRAIN_DATA}" \
    data.val_data="${EVAL_DATA}" \
    trainer.policy.model.path="${MODEL}" \
    generator.inference_engine.served_model_name="${SERVED_MODEL_NAME}" \
    trainer.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    generator.sampling_params.max_generate_length="${MAX_GENERATE_LENGTH}" \
    trainer.algorithm.max_seq_len="${MAX_MODEL_LEN}" \
    \
    trainer.export_path="${EXPORTS_DIR}" \
    trainer.ckpt_path="${CKPTS_DIR}" \
    trainer.log_path="${LOG_DIR}" \
    trainer.resume_mode=null \
    \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.loss_reduction="${LOSS_REDUCTION}" \
    trainer.algorithm.grpo_norm_by_std="${GRPO_NORM_BY_STD}" \
    trainer.algorithm.use_kl_loss="${USE_KL_LOSS}" \
    trainer.algorithm.use_kl_in_reward=false \
    trainer.update_epochs_per_batch=1 \
    trainer.epochs="${EPOCHS}" \
    trainer.eval_batch_size=8 \
    trainer.eval_before_train="${EVAL_BEFORE_TRAIN}" \
    trainer.eval_interval="${EVAL_INTERVAL}" \
    trainer.train_batch_size="${TRAIN_BATCH_SIZE}" \
    trainer.policy_mini_batch_size="${MINI_BATCH_SIZE}" \
    trainer.micro_forward_batch_size_per_gpu="${MICRO_FWD_BATCH_PER_GPU}" \
    trainer.micro_train_batch_size_per_gpu="${MICRO_TRAIN_BATCH_PER_GPU}" \
    trainer.policy.optimizer_config.lr="${LR}" \
    \
    trainer.strategy=fsdp2 \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_gpus_per_node="${NUM_GPUS}" \
    trainer.placement.ref_num_gpus_per_node="${NUM_GPUS}" \
    generator.inference_engine.num_engines="${NUM_GPUS}" \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.gpu_memory_utilization="${VLLM_GPU_MEM_UTIL}" \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.enforce_eager="${VLLM_ENFORCE_EAGER}" \
    generator.inference_engine.engine_init_kwargs.chat_template="${CHAT_TEMPLATE_PATH}" \
    generator.inference_engine.engine_init_kwargs.max_model_len="${MAX_MODEL_LEN}" \
    generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
    generator.batched=false \
    generator.step_wise_trajectories=true \
    generator.merge_stepwise_output=true \
    generator.n_samples_per_prompt="${N_SAMPLES_PER_PROMPT}" \
    generator.eval_n_samples_per_prompt=1 \
    generator.apply_overlong_filtering="${APPLY_OVERLONG_FILTERING}" \
    generator.rate_limit.enabled="${ENABLE_RATE_LIMITING}" \
    generator.rate_limit.trajectories_per_second="${TRAJECTORIES_PER_SECOND}" \
    generator.rate_limit.max_concurrency="${MAX_CONCURRENCY}" \
    \
    trainer.arctic_rl.colocate="${COLOCATE}" \
    trainer.arctic_rl.zero_stage="${ZERO_STAGE}" \
    trainer.arctic_rl.use_liger="${USE_LIGER}" \
    trainer.arctic_rl.use_arctic_inference="${USE_ARCTIC_INFERENCE}" \
    trainer.arctic_rl.use_zorro="${USE_ZORRO}" \
    trainer.arctic_rl.cuda_ipc_weight_sync="${CUDA_IPC_WEIGHT_SYNC}" \
    trainer.arctic_rl.low_memory_weight_sync="${LOW_MEM_WEIGHT_SYNC}" \
    trainer.arctic_rl.attn_implementation="${ATTN_IMPL}" \
    trainer.arctic_rl.vllm_enforce_eager="${VLLM_ENFORCE_EAGER}" \
    trainer.arctic_rl.vllm_max_model_len="${MAX_MODEL_LEN}" \
    trainer.arctic_rl.vllm_enable_prefix_caching=true \
    \
    harbor_trial_config.trials_dir="${TRIALS_DIR}" \
    harbor_trial_config.agent.kwargs.model_info.max_input_tokens="${MAX_PROMPT_LENGTH}" \
    harbor_trial_config.agent.kwargs.model_info.max_output_tokens="${MAX_GENERATE_LENGTH}" \
    \
    trainer.logger=wandb \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.run_name="${RUN_NAME}" \
    \
    "$@"
