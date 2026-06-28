#!/usr/bin/env bash
# Fast-iter smoke variant of run_bird_grpo_1.7b_8gpu.sh.
#
# Same model, same backend, same wire protocol, same env / WandB project
# — only the batch and rollout shape are shrunk so each iteration spends
# ~30s in generate instead of ~7-8min. Use this when iterating on the
# Arctic-RL wire-protocol bridge (`integrations/arctic_rl/trainer.py`)
# — every wire-shape transformation, meta dict key, post-processor
# chain, and verl_grpo loss-config path is exercised identically to the
# full BIRD recipe.
#
# When step 1 metrics emit cleanly here, switch back to
# `run_bird_grpo_1.7b_8gpu.sh` for the actual 1:1 verl PR #6 comparison.
#
# Knob deltas vs. the full recipe:
#   TRAIN_BSZ      32  -> 4     (4 prompts/batch; trajectories = 4 * N_SAMPLES = 16)
#   MINI_BSZ       32  -> 4     (single mini-batch -> grad_accum_steps=1)
#   N_SAMPLES      16  -> 4     (matches verl's GRPO group convention; smaller group)
#   PROMPT_LEN  32768  -> 4096  (BIRD prompts vary; 4K covers most without truncation)
#   RESPONSE_LEN 4096  -> 512   (caps rollout time; verl_grpo invariants are shape-
#                                independent so step-1 ppo_kl=0, clipfrac=0 still hold)

set -euxo pipefail

SKYRL_DIR=${SKYRL_DIR:-$(cd "$(dirname "$0")"/../../.. && pwd)}
DATA_DIR=${DATA_DIR:-"$HOME/data/bird"}

# Driver (same shape as flash_rl/harbor; see run_gsm8k_grpo_4gpu.sh for details).
FLASH_ATTN_WHL="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
DRIVER=(uv run --isolated --extra skyrl-train
    --with arctic-platform
    --with 'arctic-inference[vllm]'
    --with liger-kernel
    --with 'transformers==4.57.6'
    --with "flash-attn@${FLASH_ATTN_WHL}"
    -- python)

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export VLLM_LOGGING_LEVEL=WARNING  # quieter than full BIRD recipe for log readability
export ARCTIC_CUDA_IPC_LOW_MEM=0
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0

# Smoke runs are local — don't pollute the production WandB project. Disable
# logger entirely so we read step-1 metrics from stdout only (faster + no
# rate-limit risk during rapid iteration).
WANDB_LOGGER=${WANDB_LOGGER:-console}

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
MODEL=${MODEL:-"Qwen/Qwen3-1.7B"}
EXPERIMENT_NAME=skyrl_bird_grpo_smoke_${MODEL##*/}_${RUN_TS}
CHECKPOINT_DIR=${HOME}/ckpts/${EXPERIMENT_NAME}
mkdir -p "${CHECKPOINT_DIR}"

NUM_GPUS=${NUM_GPUS:-8}

# Smoke batch math (validate_batch_sizes):
#   train_per_gpu = TRAIN_BSZ * N_SAMPLES / NUM_GPUS = 4 * 4 / 8 = 2
#   mini_per_gpu  = MINI_BSZ  * N_SAMPLES / NUM_GPUS = 4 * 4 / 8 = 2
#   grad_accum    = 2 / 2 = 1
TRAIN_BSZ=${TRAIN_BSZ:-4}
MINI_BSZ=${MINI_BSZ:-4}
N_SAMPLES=${N_SAMPLES:-4}
LR=${LR:-2e-6}
PROMPT_LEN=${PROMPT_LEN:-4096}
RESPONSE_LEN=${RESPONSE_LEN:-512}

cd "${SKYRL_DIR}"

"${DRIVER[@]}" -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
    trainer.arctic_rl.colocate=true \
    trainer.arctic_rl.zero_stage=3 \
    trainer.arctic_rl.offload_optimizer=true \
    trainer.arctic_rl.log_prob_gpus=0 \
    trainer.arctic_rl.server_logs=true \
    trainer.arctic_rl.startup_timeout=1800 \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/val.parquet']" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path="${MODEL}" \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_gpus_per_node=${NUM_GPUS} \
    trainer.placement.policy_num_nodes=1 \
    generator.inference_engine.num_engines=${NUM_GPUS} \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.gpu_memory_utilization=0.5 \
    generator.inference_engine.async_engine=true \
    generator.batched=true \
    trainer.epochs=1 \
    trainer.eval_batch_size=${TRAIN_BSZ} \
    trainer.eval_before_train=false \
    trainer.eval_interval=10 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=${TRAIN_BSZ} \
    trainer.policy_mini_batch_size=${MINI_BSZ} \
    trainer.max_prompt_length=${PROMPT_LEN} \
    generator.sampling_params.max_generate_length=${RESPONSE_LEN} \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    trainer.policy.optimizer_config.lr=${LR} \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false \
    environment.env_class=bird \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    trainer.logger=${WANDB_LOGGER} \
    trainer.resume_mode=null \
    trainer.log_path="${CHECKPOINT_DIR}/logs" \
    trainer.ckpt_path="${CHECKPOINT_DIR}/ckpt" \
    trainer.ckpt_interval=-1 \
    "$@"
