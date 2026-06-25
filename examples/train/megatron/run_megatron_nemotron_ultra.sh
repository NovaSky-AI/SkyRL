set -x

# Colocated GRPO training+generation for NVIDIA-Nemotron-3-Ultra-550B-A55B on GSM8K with Megatron.
# Runs on 8 nodes of 8xH200-141GB (64 GPUs), EFA interconnect.
#
# This is *full-finetuning* RL (no LoRA, no ref/KL model). It builds on the configs proven by
# the logprob round-trip test (tests/.../gpu_ci/megatron/test_megatron_models.py::[nemotron3-ultra])
# but the test was forward-only (inference_only_init), so it had no optimizer/grads. Training adds
# bf16 grads (~same size as the weights) + the AdamW master state, so to fit the 141 GiB H200 we:
#   (a) shard depth with PP=4 (halves per-GPU weights AND grads vs the test's PP=2 -> ~34+34 GiB),
#   (b) CPU-offload the optimizer (fp32 master + Adam moments live on host RAM, not GPU),
#   (c) recompute activations, (d) bin-pack microbatches by token count, and
#   (e) drop the KL/ref model (no second 550B copy).
# VALIDATED working: this exact config trains end-to-end (reward ~0.9, gsm8k eval ~0.94, grad_norm>0).
#
# NOTE on correctness: getting coherent generations required two SkyRL fixes that are now in-tree
# (not knobs here): (1) the CUDA-IPC weight-sync sends per-GPU slicing metadata so each vLLM worker
# slices its own packed buffer correctly -- without it, weight sync corrupts vLLM at PP>2 / EP>16
# (the policy stays fine, but vLLM generates token-salad and reward stays 0); (2) a vLLM
# layerwise-reload patch (cf. vllm-project/vllm#44814) so the NemotronH Mamba `mixer.D` isn't
# dropped during reload. If you run on a SkyRL/vLLM without these, expect garbage generations.
#
# Prereqs:
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   Stage the model on every node's local disk (1.1TB; /home is too small):
#     see the staging helper used for the test (HF_HOME=/mnt/local_storage/hf_cache).
#     IMPORTANT: stage chat_template.jinja too (include *.jinja in allow_patterns). It is
#     the model's official ChatML+reasoning template; without it the tokenizer/vLLM have NO
#     chat template, the instruct model is prompted off-distribution, and reward stays 0.
#   export WANDB_API_KEY=<your_key_here>
#   bash examples/train/megatron/run_megatron_nemotron_ultra.sh

# ---------------------------------------------------------------------------
# Environment (must reach the Ray workers). These mirror the test's run env.
# ---------------------------------------------------------------------------
# Model is staged on each node's large local disk (1.1TB won't fit /home/ray's 255GB).
export HF_HOME=${HF_HOME:-/mnt/local_storage/hf_cache}
# Redirect ALL caches off the small home disk (255GB) to the big local disk (28TB).
# Workers write uv build envs, Triton/Inductor/vLLM/FlashInfer JIT caches, etc. to
# ~/.cache by default; on a small home disk that fills up and takes the node down.
# These are forwarded to Ray workers by prepare_runtime_environment.
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/mnt/local_storage/.cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/local_storage/.cache/uv}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/mnt/local_storage/.cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/local_storage/.cache/inductor}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/mnt/local_storage/.cache/vllm}
# Use the local cache only (avoids re-downloading / HF rate limits). Unset if you
# want to allow downloads on first run.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# EFA: NCCL must see the EFA libs, and SkyRL must forward LD_LIBRARY_PATH to Ray workers.
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
# vLLM multi-node executor: the default Ray compiled-DAG (shm channel) crashes the raylet
# on the cross-node hop; the V2 (MultiprocExecutor/MessageQueue) backend avoids it.
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
# Megatron attention backend (TE flash attn off; see .claude/docs/backends/megatron.md).
export NVTE_FLASH_ATTN=0
# 8-node uv cache warmup + 550B load can exceed the default placement-group timeout.
export SKYRL_RAY_PG_TIMEOUT_IN_S=${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}
# The 550B vLLM engines take a while to come up; raise the health-wait timeout.
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=${SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S:-2400}
# Set HF_TOKEN (e.g. `export HF_TOKEN=$(cat ~/.HF_TOKEN)`) for fast authenticated staging.
# HF_HUB_OFFLINE=0 (instead of 1) makes workers re-download a missing shard to the big
# disk if a node churns in un-staged, instead of erroring; with a stable staged pool, 1 is fine.
# Surface vLLM/worker logs to stdout (helpful while bringing this up; comment out later).
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=${SKYRL_DUMP_INFRA_LOG_TO_STDOUT:-1}
# export NCCL_DEBUG=WARN

# Data must be present on ALL nodes (node-local) for multi-node training. gsm8k is tiny;
# stage it to each node's local disk (e.g. copy $HOME/data/gsm8k -> here on every node).
DATA_DIR="/mnt/local_storage/data/gsm8k"
LOGGER="wandb"  # change to "console" to print to stdout
MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"

INFERENCE_BACKEND="vllm" # currently only vllm is supported for megatron

NUM_NODES=8
NUM_GPUS=8  # per node

### Megatron (policy) parallelism. world = TP*PP*DP = 64.
# TP within the NVLink domain; NemotronH Mamba requires TP | n_groups(=8), so TP in {1,2,4,8}.
MEGATRON_TP=8
# PP=4 (vs the forward-only test's 2): training adds bf16 grads (~same size as weights), so at
# EP=16/PP=2 the ~69 GiB weights/GPU + ~69 GiB grads ~= 138 GiB doesn't fit the 141 GiB H200.
# PP=4 halves the layers (hence weights AND grads) per GPU to ~34+34 GiB, which fits.
MEGATRON_PP=4
MEGATRON_CP=1
# EP=16, ETP=1 -> EDP=1 (world = TP*PP*DP = 8*4*2 = 64; EP*ETP = 16 = TP*DP = 8*2).
# This is the validated config. Earlier runs at EP=32 produced garbage vLLM generations, but that
# was the CUDA-IPC weight-sync TRANSPORT bug (rank-0 slicing metadata reused for every GPU's
# divergent buffer), NOT the expert sharding itself -- the bridge's expert export is bit-correct at
# every EP. With the per-GPU-metadata fix now in-tree, any valid EP syncs correctly, so EP is purely
# a memory/throughput knob: e.g. EP=32 (16 experts/rank vs 32) further lowers per-GPU expert memory
# if you need more headroom. EP must divide TP*DP.
MEGATRON_EP=16
MEGATRON_ETP=1

# Activation recompute (gated by trainer.gradient_checkpointing=true, which is the default).
RECOMPUTE_GRANULARITY="full"
RECOMPUTE_METHOD="uniform"
RECOMPUTE_NUM_LAYERS=1

# CPU-offload the optimizer (fp32 master + AdamW) so it doesn't sit on the GPU.
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# Bin-pack microbatches by token count (with remove_microbatch_padding). When >0,
# micro_*_batch_size_per_gpu are ignored. Bounds activation memory; a single sequence
# longer than this still gets its own microbatch. longest seq here ~= 512+1024.
MAX_TOKENS_PER_MICROBATCH=4096

### Inference engine (vLLM), colocated over the same 64 GPUs.
# TP=8 (intra-node, NVLink) x PP=4 (cross-node, EFA) = 32 GPUs/engine, 2 engines -> 64 GPUs.
# vLLM TP must divide Mamba n_groups(=8); cross-node scale comes from PP. PP=4 (not 2) keeps
# vLLM weights ~34GB/GPU so during the colocated weight sync (vLLM woken alongside the resident
# policy shard) both fit on the 141 GiB H200 (PP=2 -> ~69+69 OOMs).
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP=8
INFERENCE_ENGINE_PP=4
# Cap context: the model's native max is huge and vLLM sizes the KV pool for 1 max-len request.
INFERENCE_ENGINE_MAX_MODEL_LEN=4096
# Nemotron-3-Ultra is a REASONING model: its official chat_template.jinja defaults to
# enable_thinking=true, so each rollout emits a <think>...</think> block before the answer.
# In batched mode chat templating is done server-side by vLLM (chat_template_kwargs is not
# supported), so we cannot disable thinking from here -- instead we give generation enough
# budget to finish reasoning AND emit the final `#### <answer>` the gsm8k reward parser wants.
# (Earlier runs got reward=0 because the chat template wasn't staged at all -> the instruct
# model was prompted off-distribution and never produced a parseable answer.)
GEN_MAX_LEN=2048
# vLLM and the policy alternate on-GPU (sleep/wake); leave headroom for the policy shard.
GPU_MEMORY_UTILIZATION=0.6

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.pipeline_parallel_size=$INFERENCE_ENGINE_PP \
  generator.inference_engine.distributed_executor_backend=ray \
  generator.inference_engine.use_expandable_segments=true \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.transformer_config_kwargs.mtp_num_layers=0 \
  trainer.policy.megatron_config.transformer_config_kwargs.mtp_hybrid_override_pattern=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=$RECOMPUTE_GRANULARITY \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=$RECOMPUTE_METHOD \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=$RECOMPUTE_NUM_LAYERS \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.remove_microbatch_padding=true \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=$GEN_MAX_LEN \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.engine_init_kwargs.max_model_len=$INFERENCE_ENGINE_MAX_MODEL_LEN \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_nemotron_ultra" \
  trainer.run_name="gsm8k_nemotron_ultra_tp${MEGATRON_TP}_pp${MEGATRON_PP}_ep${MEGATRON_EP}" \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_interval=20 \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_nemotron_ultra_ckpt" \
  $@
