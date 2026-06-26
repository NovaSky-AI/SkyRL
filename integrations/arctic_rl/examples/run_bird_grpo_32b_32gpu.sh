#!/usr/bin/env bash
# SkyRL + Arctic RL backend: Qwen3-32B BIRD-SQL GRPO — 4 nodes / 32 H200s.
#
# Topology: NUM_NODES=4, GPUS_PER_NODE=8 (DP=32). vLLM TP=4, num_engines=8.
# train_batch=128 prompts x n_samples=16 = 2048 trajectories/step.
# Sibling: run_bird_grpo_32b_32gpu_fsdp.sh (same recipe, SkyRL FSDP-native).
#
# Prereq: 4-node ray cluster up; `ray status` shows 32/32 GPU.

set -euxo pipefail

SKYRL_DIR=${SKYRL_DIR:-$(cd "$(dirname "$0")"/../../.. && pwd)}
DATA_DIR=${DATA_DIR:-"$HOME/data/bird"}
PYBIN=${PYBIN:-python}
# FlashAttention impl: flash_attention_2 (broadly available) or flash_attention_3
# (Hopper-only, build from source from Dao-AILab/flash-attention hopper subdir).
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Default to ONLINE (auto-download missing models). Set OFFLINE=1 to disable
# (e.g. on isolated clusters where the model is pre-staged in HF_HOME).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
# Disable torch.inductor's on-disk cache too. Without this, a prior run that
# baked in `flashinfer_trtllm_fused_allreduce_norm` (when fuse_allreduce_rms
# was on) reuses that compiled graph and asserts during warm-up:
#   AssertionError: Flashinfer workspace must be initialized when using flashinfer
# VLLM_DISABLE_COMPILE_CACHE only covers vLLM's own cache, not inductor's.
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export ARCTIC_CUDA_IPC_LOW_MEM=0
# 32B + tie_word_embeddings is False, but keep the bypass on — it's a no-op
# when names match and a safety net if upstream Qwen3 adds new tied buffers.
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0
# verl 32B recipe ships this; helps with the 32B optimizer-state CPU offload churn.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# WandB — set WANDB_API_KEY in your environment to enable logging; override
# WANDB_PROJECT to write to a different project.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl}"
export WANDB_DISABLE_CODE=True

# Model: pass the HF id by default — transformers/vLLM auto-download to
# HF_HOME on first use. If you've pre-staged the snapshot (multi-node
# shared cache), set MODEL=<absolute snapshot path> to skip the hub lookup.
MODEL="${MODEL:-Qwen/Qwen3-32B}"
echo "MODEL=${MODEL}"

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME=skyrl_bird_grpo_Qwen3-32B_arctic_zorro_4node_${RUN_TS}
# CHECKPOINT_DIR should be on a shared filesystem visible to all nodes
# (head writes weight-sync tensor, all nodes mmap-read it).
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${HOME}/skyrl-runs/ckpts/${EXPERIMENT_NAME}}
mkdir -p "${CHECKPOINT_DIR}"

# BIRD-SQL parquets are bring-your-own (no public prep script).
if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
    echo "ERROR: BIRD-SQL parquets not found at ${DATA_DIR}/{train,val}.parquet"
    echo "       Stage your own BIRD-SQL train/val parquets and set DATA_DIR."
    exit 1
fi

NUM_NODES=4
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))   # = 32

# Global batch: 128 prompts x 16 samples = 2048 trajectories. With DP=32 and
# ulysses_sp=1: per-DP mini=64, ZoRRo micro=16 (n_samples), grad_accum=4.
TRAIN_BSZ=128
MINI_BSZ=128
N_SAMPLES=16

LR=2e-6
PROMPT_LEN=32768
RESPONSE_LEN=4096

# vLLM sampling TP — verl recipe uses TP=4 (Qwen3-32B doesn't fit per-GPU at
# bf16 + 0.5 mem_util headroom on H200). 32 GPUs / TP=4 -> 8 engine replicas.
TP_SIZE=4
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

# Inference knobs forwarded to ArcticAsyncEngineArgs via
# trainer.arctic_rl.arctic_inference_config (raw passthrough):
#   FCA: forest_cascade_attn_configs={} + cudagraph_mode=PIECEWISE.
#   pass_config.fuse_allreduce_rms=false: with 8 replicas x TP=4 colocated
#        2-per-node, the two co-located replicas race for FlashInfer's
#        per-process AllReduce IPC port; the loser gets EADDRINUSE, the
#        pass marks itself disabled, but the already-emitted graph nodes
#        still call into the workspace -> assert during CUDA-graph warmup
#        (`Flashinfer workspace must be initialized when using flashinfer`).
#        Disabling the fuse pass avoids the contested port entirely.
#        Tunji's verl recipe doesn't trip this because it co-locates one
#        sampling replica per node.
#   Spec-dec: speculative_config={method: arctic, model: <path>, ...}.
#
# These nested dicts are routed to vLLM by integrations.arctic_rl.config,
# which round-trips them through OmegaConf.to_container so AsyncEngineArgs
# sees plain dicts (the nested-override hook uses `isinstance(_, dict)`,
# which is False for omegaconf.DictConfig and silently drops the value).
# Same idiom as arctic-verl/verl/workers/remote_client/arctic_rl.py.
#
# KNOWN ISSUE (open, see follow-up): the OmegaConf round-trip alone is
# *not* sufficient end-to-end in this checkout — the nested
# compilation_config is still being dropped somewhere between
# ArcticRLClientConfig and AsyncEngineArgs.__post_init__ (vLLM resolves
# cudagraph_mode=FULL_AND_PIECEWISE and fuse_allreduce_rms=True at engine
# init, even with this override set). Until that plumbing is traced and
# fixed end-to-end, we pin optimization_level=1 below — that hard-codes
# fuse_allreduce_rms=false inside vLLM, which empirically reproduced the
# Jun 24 (skyrl_v1) 2x speedup baseline and is what unblocks TP>1 + Hopper
# from the FlashInfer-workspace assertion.
#
# Flow-style dict values need a space after every `:` — OmegaConf.from_cli
# runs yaml.load on each rhs (Hydra's CLI parser is more lenient).
USE_FCA=${USE_FCA:-True}
SPEC_MODEL=${SPEC_MODEL:-/data-fast/qwen3-32b-bird-4096-3head}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}

AI_CFG_PARTS=()
if [[ "${USE_FCA}" == "True" ]]; then
    AI_CFG_PARTS+=('forest_cascade_attn_configs: "{}"')
    AI_CFG_PARTS+=('optimization_level: 1')
    AI_CFG_PARTS+=('compilation_config: {cudagraph_mode: PIECEWISE, pass_config: {fuse_allreduce_rms: false}}')
fi
if [[ -n "${SPEC_MODEL}" && -d "${SPEC_MODEL}" ]]; then
    AI_CFG_PARTS+=("speculative_config: {method: arctic, model: ${SPEC_MODEL}, num_speculative_tokens: ${NUM_SPEC_TOKENS}}")
fi

AI_CFG_OVERRIDE=()
if (( ${#AI_CFG_PARTS[@]} > 0 )); then
    IFS=, AI_CFG_BODY="${AI_CFG_PARTS[*]}" ; unset IFS
    AI_CFG_OVERRIDE+=("trainer.arctic_rl.arctic_inference_config={${AI_CFG_BODY}}")
fi

cd "${SKYRL_DIR}"

"${PYBIN}" -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
    trainer.arctic_rl.colocate=true \
    trainer.arctic_rl.zero_stage=3 \
    trainer.arctic_rl.offload_optimizer=true \
    trainer.arctic_rl.offload_param=false \
    trainer.arctic_rl.log_prob_gpus=0 \
    trainer.arctic_rl.use_zorro=true \
    trainer.arctic_rl.use_liger=true \
    trainer.arctic_rl.attn_implementation=${ATTN_IMPL} \
    trainer.arctic_rl.enable_gradient_checkpointing=true \
    trainer.arctic_rl.ulysses_sequence_parallel_size=1 \
    trainer.arctic_rl.logits_optimization=memory \
    trainer.arctic_rl.cuda_ipc_weight_sync=true \
    trainer.arctic_rl.low_memory_weight_sync=true \
    trainer.arctic_rl.lr_warmup_ratio=0.05 \
    'trainer.arctic_rl.optimizer_betas=[0.9,0.95]' \
    trainer.arctic_rl.vllm_enforce_eager=false \
    trainer.arctic_rl.vllm_enable_prefix_caching=true \
    trainer.arctic_rl.vllm_max_num_batched_tokens=40960 \
    trainer.arctic_rl.vllm_max_num_seqs=256 \
    trainer.arctic_rl.use_arctic_inference=true \
    trainer.arctic_rl.server_logs=true \
    trainer.arctic_rl.startup_timeout=1800 \
    "${AI_CFG_OVERRIDE[@]}" \
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
