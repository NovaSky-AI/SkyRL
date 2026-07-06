#!/usr/bin/env bash
# SkyRL + Arctic RL backend: Qwen3-8B BIRD-SQL GRPO — 4 nodes / 32 H200s.
#
# Same recipe as run_bird_grpo_32b_32gpu.sh (TP=4, FCA,
# CUDA-IPC weight sync, ZoRRo, Liger), except:
#   - MODEL is Qwen3-8B (~16GB bf16) instead of Qwen3-32B.
#   - Speculative decoding is disabled by default: the 32B spec head
#     (/data-fast/qwen3-32b-bird-4096-3head) is architecturally tied to
#     Qwen3-32B and won't load on Qwen3-8B. Drop in an 8B-trained 3-head
#     checkpoint via SPEC_MODEL=... to re-enable.
#
# Use this as a faster (~4-5x step time) iteration target for the same TP>1
# code path the 32B run exercises.

set -euxo pipefail

SKYRL_DIR=${SKYRL_DIR:-$(cd "$(dirname "$0")"/../../.. && pwd)}
DATA_DIR=${DATA_DIR:-"$HOME/data/bird"}

# Driver (same shape as harbor; see run_gsm8k_grpo_4gpu.sh for details).
# FA3 (cp39-abi3 wheel from PyTorch's cu128 index) is the default on Hopper.
FLASH_ATTN_WHL="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
FLASH_ATTN3_WHL="https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
DRIVER=(uv run --isolated --extra skyrl-train
    --with arctic-platform
    --with 'arctic-inference[vllm]'
    --with liger-kernel
    --with 'transformers==4.57.6'
    --with "flash-attn@${FLASH_ATTN_WHL}"
    --with "flash-attn-3@${FLASH_ATTN3_WHL}"
    -- python)
# FA3 default (Hopper); set ATTN_IMPL=flash_attention_2 for A100/L40S.
ATTN_IMPL=${ATTN_IMPL:-flash_attention_3}

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
# Also disable torch.inductor's on-disk cache; see 32B launcher for the
# stale-compiled-graph rationale.
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export ARCTIC_CUDA_IPC_LOW_MEM=0
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl}"
export WANDB_DISABLE_CODE=True

MODEL="${MODEL:-Qwen/Qwen3-8B}"
echo "MODEL=${MODEL}"

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME=skyrl_bird_grpo_Qwen3-8B_arctic_zorro_4node_${RUN_TS}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${HOME}/skyrl-runs/ckpts/${EXPERIMENT_NAME}}
mkdir -p "${CHECKPOINT_DIR}"

if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
    echo "ERROR: BIRD-SQL parquets not found at ${DATA_DIR}/{train,val}.parquet"
    echo "       Stage your own BIRD-SQL train/val parquets and set DATA_DIR."
    exit 1
fi

NUM_NODES=4
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))

TRAIN_BSZ=128
MINI_BSZ=128
N_SAMPLES=16

LR=2e-6
PROMPT_LEN=32768
RESPONSE_LEN=4096

# Same TP=4 as 32B to exercise the same multi-rank flashinfer code path.
TP_SIZE=4
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

# Inference-engine optimizations are opt-in via typed knobs on
# ``trainer.arctic_rl`` — the integration auto-injects the matching raw
# vLLM kwargs (FCA, multi-replica FlashInfer workaround, Arctic
# speculative_config) so the launcher never needs a raw vllm_config block.
#   - ``use_arctic_inference=true``  -> FCA on, fuse_allreduce_rms workaround
#                                       when num_engines > 1.
#   - ``speculative_model=<path>``   -> Arctic speculative decoding on with
#                                       ``num_speculative_tokens`` draft tokens.
SPEC_MODEL=${SPEC_MODEL:-}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}

ARCTIC_SPEC_OVERRIDE=()
if [[ -n "${SPEC_MODEL}" ]]; then
    ARCTIC_SPEC_OVERRIDE+=("trainer.arctic_rl.speculative_model=${SPEC_MODEL}")
    ARCTIC_SPEC_OVERRIDE+=("trainer.arctic_rl.num_speculative_tokens=${NUM_SPEC_TOKENS}")
fi

cd "${SKYRL_DIR}"

"${DRIVER[@]}" -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
    trainer.arctic_rl.colocate=true \
    trainer.arctic_rl.zero_stage=3 \
    trainer.arctic_rl.offload_optimizer=true \
    trainer.arctic_rl.attn_implementation=${ATTN_IMPL} \
    trainer.arctic_rl.cuda_ipc_weight_sync=true \
    trainer.arctic_rl.low_memory_weight_sync=true \
    trainer.arctic_rl.lr_warmup_ratio=0.05 \
    'trainer.arctic_rl.optimizer_betas=[0.9,0.95]' \
    trainer.arctic_rl.vllm_enforce_eager=false \
    trainer.arctic_rl.vllm_max_num_batched_tokens=40960 \
    trainer.arctic_rl.server_logs=true \
    trainer.arctic_rl.startup_timeout=1800 \
    "${ARCTIC_SPEC_OVERRIDE[@]}" \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/val.parquet']" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path="${MODEL}" \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_gpus_per_node=${GPUS_PER_NODE} \
    trainer.placement.policy_num_nodes=${NUM_NODES} \
    generator.inference_engine.num_engines=${NUM_ENGINES} \
    generator.inference_engine.tensor_parallel_size=${TP_SIZE} \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.gpu_memory_utilization=0.5 \
    generator.inference_engine.async_engine=true \
    generator.batched=true \
    trainer.epochs=1 \
    trainer.eval_batch_size=32 \
    trainer.eval_before_train=false \
    trainer.eval_interval=100 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=${TRAIN_BSZ} \
    trainer.policy_mini_batch_size=${MINI_BSZ} \
    trainer.max_prompt_length=${PROMPT_LEN} \
    generator.sampling_params.max_generate_length=${RESPONSE_LEN} \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    generator.eval_sampling_params.max_generate_length=${RESPONSE_LEN} \
    generator.eval_sampling_params.temperature=0.0 \
    generator.eval_sampling_params.top_p=1.0 \
    generator.eval_sampling_params.top_k=-1 \
    generator.eval_n_samples_per_prompt=1 \
    trainer.policy.optimizer_config.lr=${LR} \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false \
    environment.env_class=bird \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    trainer.logger=${LOGGER:-wandb} \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CHECKPOINT_DIR}/logs" \
    trainer.ckpt_path="${CHECKPOINT_DIR}/ckpt" \
    trainer.ckpt_interval=-1 \
    "$@"
