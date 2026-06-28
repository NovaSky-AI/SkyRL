#!/usr/bin/env bash
# Native SkyRL FSDP backend: Qwen3-32B BIRD-SQL GRPO recipe — 4 nodes / 32 H200s.
#
# Sibling of run_bird_grpo_32b_32gpu.sh: SAME model, SAME data, SAME hyperparams,
# SAME placement, SAME WandB project — only the training backend differs.
# Apples-to-apples E2E time-per-step comparison vs the Arctic RL backend.
#
# Differences vs the arctic launcher:
#   - no trainer.override_entrypoint (default core FSDP path; drop all arctic_rl flags)
#   - generator stays vLLM, no ArcticInference (no FCA, no speculative decoding)
#   - trainer.placement.colocate_all=true (native SkyRL colocation, not Arctic's)
#   - No CUDA-IPC weight sync (uses SkyRL's native FSDP weight sync)
#
# Same prereqs: 4-node ray cluster up, HF Qwen3-32B downloaded.

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
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Liger off: Qwen3 Liger kernel hits a Triton illegal-mem-access on packed-seq
# inputs (cu_seqlens variable, attention_mask=None). Compensate with micro=2
# below so LM-head logits fit on H200.
export SKYRL_USE_LIGER=0

# Same W&B project as the arctic launcher so the two runs sit side-by-side.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl}"
export WANDB_DISABLE_CODE=True

MODEL="${MODEL:-Qwen/Qwen3-32B}"
echo "MODEL=${MODEL}"

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME=skyrl_bird_grpo_Qwen3-32B_fsdp_4node_${RUN_TS}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${HOME}/skyrl-runs/ckpts/${EXPERIMENT_NAME}}
mkdir -p "${CHECKPOINT_DIR}"

NUM_NODES=4
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Same batch math as the arctic launcher.
TRAIN_BSZ=128
MINI_BSZ=128
N_SAMPLES=16
LR=2e-6
PROMPT_LEN=32768
RESPONSE_LEN=4096

TP_SIZE=4
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

cd "${SKYRL_DIR}"

FSDP_BIRD_ENTRY="${SKYRL_DIR}/integrations/arctic_rl/examples/fsdp_bird_entry.py"

"${DRIVER[@]}" "${FSDP_BIRD_ENTRY}" \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/val.parquet']" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path="${MODEL}" \
    trainer.strategy=fsdp2 \
    trainer.placement.colocate_all=true \
    trainer.placement.policy_num_gpus_per_node=${GPUS_PER_NODE} \
    trainer.placement.policy_num_nodes=${NUM_NODES} \
    trainer.policy.fsdp_config.cpu_offload=false \
    trainer.policy.fsdp_config.reshard_after_forward=true \
    trainer.policy.optimizer_config.offload_after_step=true \
    trainer.policy.sequence_parallel_size=1 \
    trainer.flash_attn=true \
    trainer.micro_train_batch_size_per_gpu=${MICRO_TRAIN:-2} \
    trainer.micro_forward_batch_size_per_gpu=${MICRO_FWD:-2} \
    trainer.use_sample_packing=true \
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
    trainer.logger=wandb \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CHECKPOINT_DIR}/logs" \
    trainer.ckpt_path="${CHECKPOINT_DIR}/ckpt" \
    trainer.ckpt_interval=-1 \
    "$@"
