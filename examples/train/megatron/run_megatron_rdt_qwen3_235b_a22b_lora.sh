set -x

# Disaggregated GRPO for Qwen3-235B-A22B (LoRA) on GSM8K with Megatron and
# RDT (Ray Direct Transport / NIXL) sharded weight sync.
# Runs on 3 nodes of 8xH100s: 2 trainer nodes + 1 inference node.
#
# This is the at-scale RDT weight-sync demonstration: LoRA is merged into the
# base weights at every sync, so each sync moves the FULL ~470GB merged model
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
# Optional RDT tuning (defaults shown): SKYRL_RDT_NUM_BUFFERS=2 (publish ring
# depth), SKYRL_RDT_LOOKAHEAD=2 (gather lookahead credits),
# SKYRL_RDT_PP_LOCAL=1 (each pipeline stage gathers and serves only its own
# layers; engages at pp>1 and reverts to gather-to-all on its own if a group
# spans two stages -- set 0 to force the old path).

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

# LoRA configuration (merged into full weights at each sync)
LORA_RANK=128
LORA_ALPHA=128

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
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
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
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.algorithm.use_kl_loss=false \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/rdt_qwen3_235b_a22b_lora" \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyrl-rdt" \
  trainer.run_name="rdt_qwen3_235b_a22b_lora_tp${MEGATRON_TP}pp${MEGATRON_PP}ep${MEGATRON_EP}" \
  $@
