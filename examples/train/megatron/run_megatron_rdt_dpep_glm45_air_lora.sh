set -x

# Disaggregated GRPO for GLM-4.5-Air (LoRA) on GSM8K with Megatron and RDT
# (Ray Direct Transport / NIXL) sharded weight sync — vLLM in DP8/TP1 with
# expert parallelism (EP8).
# Runs on 2 nodes of 8xH100s: 1 trainer node + 1 inference node.
#
# GLM-4.5-Air (zai-org/GLM-4.5-Air) is a 106B-total / 12B-active MoE:
# 46 layers (layer 0 dense), 128 routed experts (1408 moe-intermediate,
# top-8, DeepSeek-style sigmoid routing with expert bias) + 1 shared expert
# per MoE layer, GQA 96Q/8KV heads, untied 151k vocab — ~214 GB in bf16.
# The checkpoint's single MTP (nextn) layer is not loaded by either side
# (no mtp_num_layers on the trainer, no speculative config on vLLM), so it
# is simply absent from the sync metadata.
#
# Parallelism (both sides shard the same way, which is what makes the sync
# cheap):
#   * Trainer: TP=1/PP=1/EP=8 (MoE TP is sized by ACTIVE params — 12B needs
#     no TP; EP8 puts 16 full experts on each of the 8 ranks, DP=8 for the
#     LoRA training itself). Each rank holds ~25 GB of experts + ~14 GB of
#     replicated non-expert weights.
#   * vLLM: DP=8/TP=1 with expert parallel (EP size = TP x DP = 8), matching
#     the trainer's EP8 so each consumer's 16-expert range maps to exactly
#     one trainer coordinate: ~2 chunks per layer group (1 expert chunk + 1
#     replicated chunk) per consumer instead of a TP slice of every
#     coordinate. Same DP+EP shape that measured 3.2-3.3s engine-level syncs
#     on Qwen3-235B; this model moves ~214 GB/sync (LoRA merged into the
#     full weights at every sync).
#
# Memory notes (80 GB cards): at ~40 GB/rank of vLLM weights CUDA graphs stay
# ON, but gpu_memory_utilization must be 0.75, NOT higher. The audited ledger
# from allocator snapshots on this exact config: NON-torch memory (CUDA
# context, NCCL, EFA/NIXL internals, the driver side of CUDA-graph capture)
# runs ~10 GB/GPU on a DP+EP worker, so torch's usable pool is ~69 GB; the
# sync's transient churn keeps ~5 GB of split-segment reservations on top of
# active memory; so the ACTIVE budget (weights 36.7 + KV + graphs 3.1 + MoE
# workspace + 2.5 GB NIXL receive arenas) must stay under ~63 GB. vLLM fills
# its whole fraction with KV no matter how small the model, so the fraction
# is the lever: 0.75 -> ~13.7 GB KV and ~4 GB of true slack. (0.80/0.85/0.90
# all OOM'd the first sync's layer materialization.) Two more knobs are
# deliberate:
#   * max_num_batched_tokens=2048 — the DP+EP fused-MoE workspace scales
#     with tokens x dp_size (~2 GiB here; the 8192 default would want ~8).
#   * engine_init_kwargs.max_model_len=2048 — the KV cache must fit one
#     max-model-len request at engine start; the checkpoint default is 131K.
#     This recipe generates 512 prompt + 1024 completion tokens.
# The RDT receive arenas (2 ring slots x 1.25 GiB — a slot holds the largest
# chunk, the full untied embed/lm_head matrix) live OUTSIDE vLLM's memory
# fraction and fit in the ~8 GB the 0.85 utilization leaves free.
# (use_expandable_segments does NOT work here: NIXL/RDMA cannot register
# VMM-backed allocations — the arena registration fails at init. The sync's
# allocator-fragmentation pressure is instead handled by the engine's bounded
# retirement queue, which caps materialized-layer churn at ~2 groups.)
#
# gather_lookahead stays at the default 1: at most TWO layer groups resident
# on the trainer at once — the memory contract that scales to larger models.
#
# Requirements: same as run_megatron_rdt_qwen3_235b_a22b_lora.sh
# (disaggregated placement, cross-node NIXL / aws-efa-installer >= 1.47,
# DATA_DIR and HF_HOME on shared storage), plus the weights:
#   HF_HUB_ENABLE_HF_TRANSFER=1 hf download zai-org/GLM-4.5-Air
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/megatron/run_megatron_rdt_dpep_glm45_air_lora.sh

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout
: "${NUM_STEPS:=15}" # set to null to train a full epoch
: "${WEIGHT_SYNC_BACKEND:=sharded_rdt}"

MODEL_NAME="zai-org/GLM-4.5-Air"
INFERENCE_BACKEND="vllm" # currently only vllm is supported for megatron

# 1 dedicated trainer node
NUM_NODES=1
NUM_GPUS=8

# MoE: no TP (12B active), EP8 -> 16 experts/rank, DP=8 for training
MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

# GLM-4.5-Air routes DeepSeek-V3 style: sigmoid scores + expert bias. No
# load-balancing loss during RL; grouped GEMM for the 128-expert layers.
MOE_TOKEN_DISPATCHER="alltoall"
MOE_ROUTER_LB="none"
MOE_GROUPED_GEMM=true
MOE_ROUTER_SCORE_FN="sigmoid"
MOE_ROUTER_EXPERT_BIAS=true

# 1 dedicated inference node: one engine of 8 data-parallel, expert-parallel
# single-GPU ranks (EP size = TP x DP = 8, matching the trainer's EP8).
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=1
INFERENCE_ENGINE_DP=8
INFERENCE_ENGINE_EP=8

# LoRA configuration (merged into full weights at each sync)
LORA_RANK=128
LORA_ALPHA=128

export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=3600

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.strategy=megatron \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.data_parallel_size=$INFERENCE_ENGINE_DP \
  generator.inference_engine.expert_parallel_size=$INFERENCE_ENGINE_EP \
  generator.inference_engine.gpu_memory_utilization=0.75 \
  generator.inference_engine.max_num_batched_tokens=2048 \
  generator.inference_engine.engine_init_kwargs.max_model_len=2048 \
  generator.inference_engine.weight_sync_backend=$WEIGHT_SYNC_BACKEND \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER \
  trainer.policy.megatron_config.moe_router_load_balancing_type=$MOE_ROUTER_LB \
  trainer.policy.megatron_config.moe_grouped_gemm=$MOE_GROUPED_GEMM \
  trainer.policy.megatron_config.moe_router_score_function=$MOE_ROUTER_SCORE_FN \
  trainer.policy.megatron_config.moe_router_enable_expert_bias=$MOE_ROUTER_EXPERT_BIAS \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true \
  trainer.remove_microbatch_padding=true \
  trainer.epochs=1 \
  trainer.max_training_steps=$NUM_STEPS \
  trainer.eval_before_train=false \
  trainer.eval_interval=1000 \
  trainer.ckpt_interval=1000 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  generator.n_samples_per_prompt=4 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.algorithm.use_kl_loss=false \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/rdt_dpep_glm45_air_lora" \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyrl-rdt" \
  trainer.run_name="rdt_dpep_glm45_air_lora_tp${MEGATRON_TP}pp${MEGATRON_PP}ep${MEGATRON_EP}" \
  $@
