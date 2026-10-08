set -x

# Disaggregated GRPO for Qwen3-235B-A22B (full fine-tuning) on GSM8K with Megatron and RDT
# (Ray Direct Transport / NIXL) sharded weight sync — vLLM in DP16/TP1 with
# expert parallelism (EP16) spread across TWO inference nodes, with CUDA graphs
# and torch.compile ENABLED.
# Runs on 4 nodes of 8xH100s: 2 trainer nodes + 2 inference nodes.
#
# This is the 4-node variant of run_megatron_rdt_dpep_qwen3_235b_a22b_lora.sh.
# The extra inference node is spent on HALVING the per-rank weight footprint,
# not on a second model replica — that is the whole point of the recipe.
#
# ---- ONE 16-WAY ENGINE, NOT TWO 8-WAY ENGINES ------------------------------
# The two obvious ways to use a second inference node are not close:
#
#   * num_engines=2, each DP8/EP8 (one per node): two full model replicas.
#     Every rank still holds the same 67.8 GiB of weights as the 3-node recipe,
#     so there is still no room for a CUDA-graph pool (see below), and each
#     replica pulls its own copy of the model every sync — ~2x the trainer's
#     egress for zero memory relief.
#   * num_engines=1, DP16/EP16 spanning both nodes (THIS RECIPE): one model,
#     sharded 16 ways. The expert share per rank halves, which is what pays for
#     CUDA graphs, a real KV cache, and headroom the 3-node recipe does not have.
#
# SkyRL provisions one server actor per DP rank (ServerGroup with
# num_servers=data_parallel_size), rank 0 publishes its node IP + RPC port as
# the DP rendezvous, and the placement group is PACK, so the 16 single-GPU
# bundles fill one node and spill onto the second. Nothing here requires the
# engine to fit in a node — that constraint only applies to colocated runs on
# the vLLM `mp` executor, and this is non-colocated on `ray`.
#
# ---- WHY enforce_eager CAN FINALLY BE FALSE --------------------------------
# The 3-node DPEP recipe sets enforce_eager=true purely because it cannot
# afford the graph pool: DP leaves attention and embeddings UNSLICED, so each
# rank holds 67.8 GiB of an 80 GiB card and ~11 GiB has to cover non-torch
# memory, activations, KV and the receive buffers. Dropping the ~2 GiB CUDA-graph
# pool was the only way to fit. That is a property of the 8-way shape, not of
# RDT: the TP8 recipe (run_megatron_rdt_qwen3_235b_a22b_lora.sh) never sets
# enforce_eager, because TP8 slices everything down to ~55 GiB/rank.
#
# At DP16/EP16 the weights are:
#
#   experts      423.0 GiB total / 16 ranks         = 26.4 GiB
#   non-expert   attention + norms + untied embeds  = 14.9 GiB  (replicated:
#                                                       DP does not shard it)
#   -------------------------------------------------------------
#   per rank                                        ~ 41.3 GiB   (was 67.8)
#
# ~26 GiB comes back per card, which is enough for compilation, the graph pool
# and a usable KV cache at the same time. The closest MEASURED point for this
# shape is the GLM-4.5-Air recipe (run_megatron_rdt_dpep_glm45_air_lora.sh):
# DP8/EP8, ~36.7 GiB of weights per rank, CUDA graphs ON, and
# gpu_memory_utilization=0.75 — with 0.80/0.85/0.90 all OOMing the first sync's
# layer materialization. This recipe sits ~4.6 GiB above that on weights and
# reuses its 0.75, which is the conservative reading of that ledger. The audited
# per-GPU ledger it comes from (DP+EP worker, 80 GB card):
#
#   non-torch (CUDA context, NCCL, EFA/NIXL internals, the driver side of
#     graph capture)                                 ~10 GiB
#   vLLM's own fraction (0.75 x ~79.6) covers weights + non-torch +
#     activation peak, and gives the remainder to KV  ~7-9 GiB of KV here
#   CUDA-graph pool, captured AFTER that accounting   ~3 GiB
#   RDT receive buffers, OUTSIDE vLLM's fraction:
#     num_rdt_buffers(2) x largest chunk (the full untied
#     151936x4096 embed/lm_head matrix, 1.16 GiB)     ~2.5 GiB
#   churn reserve for the sync's materialization      ~5 GiB
#
# Two knobs stay pinned for reasons that did not change:
#   * max_num_batched_tokens=2048 — the DP+EP fused-MoE workspace is sized for
#     every DP rank's tokens landing on one EP rank (~tokens x dp x hidden), so
#     it scales with DP: at dp16 this costs roughly twice what it cost at dp8
#     (~4 GiB), and the 8192 default is hopeless. It is charged to the
#     activation peak inside vLLM's fraction, so it comes out of KV rather than
#     OOMing; halve it to 1024 if the reported KV cache is too small to keep
#     max_num_seqs busy.
#   * engine_init_kwargs.max_model_len=2048 — the KV cache must fit one
#     max-model-len request at engine start; the checkpoint default is 40960.
#     This recipe generates 512 prompt + 1024 completion tokens.
# (use_expandable_segments must stay off on the engine: NIXL/RDMA cannot
# register VMM-backed allocations, so buffer registration fails at init.)
#
# ---- WHAT THIS DOES TO THE WEIGHT SYNC -------------------------------------
# Consumer count doubles (16 instead of 8), and that cuts both ways:
#
#   * Per-consumer pull SHAPE is unchanged. The trainer stays at EP8, so a
#     trainer coordinate holds 16 consecutive experts and each vLLM rank's 8
#     experts are a subset of exactly ONE coordinate — still ~2 chunks per layer
#     group per consumer (1 expert chunk + 1 replicated chunk), not a TP-sliced
#     pull of every coordinate.
#   * Total bytes on the wire go UP ~20%: the replicated non-expert share
#     (14.9 GiB) is pulled by 16 consumers instead of 8, so the fleet pulls
#     ~660 GiB instead of ~540 GiB. But it lands on TWO receiving nodes, so
#     per-node ingress drops from ~540 to ~330 GiB.
# Net wall-clock effect is a measurement, not a prediction. The 3-node DPEP
# reference is ~3.2-3.5 s of engine-level sync (timing/sync_weights), with
# timing/sync_weights_total ~4 s higher because of vLLM's DP pause/resume
# bracket around the sync — a control-plane cost outside the weight transfer,
# and one that does not get cheaper with more DP ranks.
#
# gather_lookahead stays at the default 1: at most TWO layer groups resident on
# the trainer at once — the memory contract that scales to larger models.
#
# ---- REQUIREMENTS ----------------------------------------------------------
# Same as run_megatron_rdt_qwen3_235b_a22b_lora.sh (disaggregated placement,
# cross-node NIXL / aws-efa-installer >= 1.47 on EVERY GPU node, weights
# prefetched to a shared HF_HOME, DATA_DIR on shared storage), plus:
#   - FOUR 8xH100 nodes, not three.
#   - The two inference nodes must be able to reach each other on the DP RPC
#     port: DP rank 0 publishes its node IP and the other 15 ranks dial it.
#   - Optional but recommended with compilation on: point VLLM_CACHE_ROOT at
#     shared storage so both inference nodes share one torch.compile cache
#     instead of each paying the compile on first start, e.g.
#     export VLLM_CACHE_ROOT=/mnt/cluster_storage/vllm_cache
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/megatron/run_megatron_rdt_dpep_qwen3_235b_a22b_4node.sh
#
# ---- WILL NOT FIT ON THE 4x8xH100 CLUSTER AS WRITTEN ----------------------
# This recipe was LoRA (frozen base + r128 adapters, merged at every sync),
# which is what made Qwen3-235B-A22B trainable on 16 GPUs at all. Full
# fine-tuning needs, per trainer GPU: 29 GB of bf16 weights + 29 GB of
# bf16 grads = 59 GB of an 80 GB card BEFORE activations, the RDT gather
# buffers (~5 GB) and allocator churn -- and the CPU-offloaded optimizer
# (fp32 master + Adam m/v = 12 B/param) needs ~1.41 TB per node against
# 1.92 TB of RAM. Both are over budget.
#
# To run this full-FT you need more trainer nodes (~64 GPUs to reach the
# same per-GPU footprint LoRA had). To run it on THIS cluster, restore LoRA:
#   trainer.policy.model.lora.rank=128 trainer.policy.model.lora.alpha=128
# merge_lora defaults to true, so the sync still streams full merged weights --
# LoRA changes the TRAINER's memory, never the weight-sync path or its timing.
# --------------------------------------------------------------------------

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout
: "${NUM_STEPS:=15}" # set to null to train a full epoch
: "${WEIGHT_SYNC_BACKEND:=sharded_rdt}"

MODEL_NAME="Qwen/Qwen3-235B-A22B"
INFERENCE_BACKEND="vllm" # currently only vllm is supported for megatron

# 2 dedicated trainer nodes
NUM_NODES=2
NUM_GPUS=8

# max TP is 4: Qwen3-235B-A22B uses Grouped Query Attention with 4 KV groups
MEGATRON_TP=4
MEGATRON_PP=2
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

# 2 dedicated inference nodes: ONE engine of 16 data-parallel, expert-parallel
# single-GPU ranks spanning both. EP size = TP x DP = 16, so each rank holds 8
# of the 128 experts — a subset of exactly one trainer EP8 coordinate.
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=1
INFERENCE_ENGINE_DP=16
INFERENCE_ENGINE_EP=16

# the 235B engine takes well over the default 600s to load weights, and with
# compilation enabled the first start also pays torch.compile on each node
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
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.max_num_batched_tokens=2048 \
  generator.inference_engine.max_num_seqs=128 \
  generator.inference_engine.engine_init_kwargs.max_model_len=2048 \
  generator.inference_engine.weight_sync_backend=$WEIGHT_SYNC_BACKEND \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
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
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/rdt_dpep_qwen3_235b_a22b_4node" \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyrl-rdt" \
  trainer.run_name="rdt_dpep_qwen3_235b_a22b_4node_tp${MEGATRON_TP}pp${MEGATRON_PP}ep${MEGATRON_EP}_dp${INFERENCE_ENGINE_DP}" \
  $@
