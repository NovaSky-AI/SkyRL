set -x

# Disaggregated GRPO for Qwen3-30B-A3B (full fine-tuning) on GSM8K with Megatron and
# RDT (Ray Direct Transport / NIXL) sharded weight sync.
# Runs on 3 nodes of 8xH100s: 2 trainer nodes + 1 inference node.
#
# Unlike the NCCL-broadcast weight sync, `sharded_rdt` has each trainer rank
# publish its shard once and the inference engine pull directly over the
# fabric (NIXL), overlapping gather/publish/pull. On this topology full-weight
# syncs of the merged 30B model take ~5s (vs ~15s+ for nccl broadcast).
#
# Requirements:
#   - sharded_rdt requires disaggregated placement (colocate_all=false).
#   - Cross-node NIXL transport. On AWS EFA clusters the nodes need
#     aws-efa-installer >= 1.47; the in-tree runtime shim selects the
#     LIBFABRIC backend automatically (no manual patching).
#   - On multi-node clusters, DATA_DIR (and HF_HOME, if set) should be on
#     shared storage — the entrypoint and workers run on arbitrary nodes.
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/megatron/run_megatron_rdt_qwen3_30b_a3b_lora.sh
#
# Optional RDT tuning (defaults shown): SKYRL_RDT_NUM_BUFFERS=2 (receive/serve
# ring depth) and SKYRL_RDT_LOOKAHEAD=1 (gathered-but-unfreed groups the
# trainer's gather loop runs ahead by; a gathered group is served immediately
# and trainer-resident memory is bounded at lookahead + 1 groups). PP-local +
# EP-local serving is always on: each pipeline stage gathers only its own
# layers and each expert-parallel rank serves only its own experts (a source
# with shared groups, e.g. tied embeddings, demotes itself to the full gather;
# SKYRL_RDT_STACKED_EXPERTS=0 forces the plain source).

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout
: "${NUM_STEPS:=15}" # set to null to train a full epoch

MODEL_NAME="Qwen/Qwen3-30B-A3B"
INFERENCE_BACKEND="vllm" # currently only vllm is supported for megatron

# sharded_rdt is the RDT weight sync path; "nccl" runs the same recipe with
# broadcast weight sync for comparison.
WEIGHT_SYNC_BACKEND="sharded_rdt"

# 2 dedicated trainer nodes
NUM_NODES=2
NUM_GPUS=8

MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

# 1 dedicated inference node
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=8

# large-model engines can take a while to load weights on first start
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
  generator.inference_engine.gpu_memory_utilization=0.85 \
  generator.inference_engine.weight_sync_backend=$WEIGHT_SYNC_BACKEND \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
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
  trainer.ckpt_path="$HOME/ckpts/rdt_qwen3_30b_a3b" \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyrl-rdt" \
  trainer.run_name="rdt_qwen3_30b_a3b_tp${MEGATRON_TP}pp${MEGATRON_PP}ep${MEGATRON_EP}" \
  $@
