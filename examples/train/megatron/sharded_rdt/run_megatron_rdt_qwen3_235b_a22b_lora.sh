set -x

# Disaggregated GRPO for Qwen3-235B-A22B (full fine-tuning) on GSM8K with Megatron and
# RDT (Ray Direct Transport / NIXL) sharded weight sync.
# Runs on 3 nodes of 8xH100s: 2 trainer nodes + 1 inference node.
#
# This is the at-scale RDT weight-sync demonstration: every sync moves the
# FULL ~470GB model
# from 16 Megatron ranks (tp4/pp2/ep8) to the vLLM TP8 engine. With
# sharded_rdt each trainer rank publishes its shard once and the engine pulls
# directly over the fabric (NIXL), overlapping gather/publish/pull: syncs take
# ~9-13s on this topology vs ~62-69s with nccl broadcast (~6.5x faster), with
# identical reward curves and rollout-vs-train logprob agreement.
#
# Requirements:
#   - sharded_rdt requires disaggregated placement (colocate_all=false).
#   - Cross-node NIXL transport. On AWS EFA clusters the nodes need
#     aws-efa-installer >= 1.47; the in-tree runtime shim selects the
#     LIBFABRIC backend automatically (no manual patching).
#   - Qwen3-235B-A22B weights (~470GB) prefetched to a shared HF_HOME, e.g.:
#     HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-235B-A22B
#   - On multi-node clusters, DATA_DIR and HF_HOME must be on shared storage —
#     the entrypoint and workers run on arbitrary nodes.
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/megatron/run_megatron_rdt_qwen3_235b_a22b_lora.sh
#
# To run the same recipe with nccl broadcast weight sync for comparison:
#   WEIGHT_SYNC_BACKEND=nccl bash ...run_megatron_rdt_qwen3_235b_a22b_lora.sh \
#     generator.inference_engine.gpu_memory_utilization=0.85
# (0.85 because SkyRL sets NCCL_CUMEM_ENABLE=0 for the nccl backend, which
# costs ~5GiB/GPU of communicator buffers on the engine; 0.95 then fails
# vLLM's startup free-memory check.)
#
# Optional RDT tuning (defaults shown): SKYRL_RDT_NUM_BUFFERS=2 (receive/serve
# ring depth) and SKYRL_RDT_LOOKAHEAD=1 (gathered-but-unfreed groups the
# trainer's gather loop runs ahead by; a gathered group is served immediately
# and trainer-resident memory is bounded at lookahead + 1 groups). PP-local +
# EP-local serving is always on: each pipeline stage gathers only its own
# layers and each expert-parallel rank serves only its own experts (a source
# with shared groups, e.g. tied embeddings, demotes itself to the full gather;
# SKYRL_RDT_STACKED_EXPERTS=0 forces the plain source).
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

# 1 dedicated inference node
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=8

# the 235B engine takes well over the default 600s to load weights
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
  generator.inference_engine.gpu_memory_utilization=0.95 \
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
  trainer.ckpt_path="$HOME/ckpts/rdt_qwen3_235b_a22b" \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyrl-rdt" \
  trainer.run_name="rdt_qwen3_235b_a22b_tp${MEGATRON_TP}pp${MEGATRON_PP}ep${MEGATRON_EP}" \
  $@
