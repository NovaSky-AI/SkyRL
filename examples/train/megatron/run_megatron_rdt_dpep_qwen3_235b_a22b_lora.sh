set -x

# Disaggregated GRPO for Qwen3-235B-A22B (LoRA) on GSM8K with Megatron and
# RDT (Ray Direct Transport / NIXL) sharded weight sync — vLLM in DP8/TP1 with
# expert parallelism (EP8) instead of TP8.
# Runs on 3 nodes of 8xH100s: 2 trainer nodes + 1 inference node.
#
# The DP+EP variant of run_megatron_rdt_qwen3_235b_a22b_lora.sh, and the
# fastest weight-sync configuration measured on this topology: each vLLM rank
# holds 16 FULL experts matching exactly one trainer EP coordinate, so a
# consumer pulls ~2 chunks per layer group (1 expert chunk + 1 replicated
# chunk) instead of 9 TP-sliced ones — 190 pulls/consumer/sync instead of 848,
# at 43 GiB/s wire instead of ~27 — for a 3.2-3.3s engine-level sync (vs ~6s
# with TP8). The driver-level sync_weights timer reads ~4s higher than the
# engine wall: that is vLLM's DP pause/resume bracket around the sync (all 8
# DP engines quiesce through the DP coordinator), a control-plane cost outside
# the weight transfer itself.
#
# ---- MEMORY: every inference-engine knob below is load-bearing on 80GB -----
# DP leaves attention/embeddings UNSLICED, so weights are 67.8 GiB/rank (vs
# ~59 under TP8), and ~11 GiB must cover everything else:
#   * max_num_batched_tokens=2048 — the DP+EP fused-MoE workspace is sized for
#     every DP rank's tokens landing on one EP rank (~tokens x dp x hidden);
#     the 8192 default wants 8 GiB and OOMs vLLM's startup profiling pass.
#   * engine_init_kwargs.max_model_len=2048 — the KV cache must fit one
#     max-model-len request; the checkpoint default (40960) needs ~7.5 GiB
#     when only ~2.8 GiB remains. This recipe generates 512+1024 tokens.
#   * enforce_eager=true + gpu_memory_utilization=0.90 — the RDT receive
#     arenas live OUTSIDE vLLM's memory fraction, and under DP their ring
#     slots hold the largest chunk, the full untied embed/lm_head matrix:
#     2 slots x 1.25 GiB (vs 256 MB slots under TP8). Dropping the ~2 GiB
#     CUDA-graph pool and 0.05 of utilization makes room. Costs decode speed
#     only; sync time is unaffected.
#
# gather_lookahead stays at the default 1: at most TWO layer groups resident
# on the trainer at once — the memory contract that scales to larger models.
#
# Requirements: same as run_megatron_rdt_qwen3_235b_a22b_lora.sh (disaggregated
# placement, cross-node NIXL / aws-efa-installer >= 1.47, weights prefetched
# to a shared HF_HOME, DATA_DIR on shared storage).
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/megatron/run_megatron_rdt_dpep_qwen3_235b_a22b_lora.sh

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

# 1 dedicated inference node: one engine of 8 data-parallel, expert-parallel
# single-GPU ranks. EP size = TP x DP = 8, matching the trainer's EP8 so each
# consumer's expert range maps to exactly one trainer coordinate.
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=1
INFERENCE_ENGINE_DP=8
INFERENCE_ENGINE_EP=8

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
  generator.inference_engine.data_parallel_size=$INFERENCE_ENGINE_DP \
  generator.inference_engine.expert_parallel_size=$INFERENCE_ENGINE_EP \
  generator.inference_engine.gpu_memory_utilization=0.90 \
  generator.inference_engine.enforce_eager=true \
  generator.inference_engine.max_num_batched_tokens=2048 \
  generator.inference_engine.max_num_seqs=128 \
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
  trainer.ckpt_path="$HOME/ckpts/rdt_dpep_qwen3_235b_a22b_lora" \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyrl-rdt" \
  trainer.run_name="rdt_dpep_qwen3_235b_a22b_lora_tp${MEGATRON_TP}pp${MEGATRON_PP}ep${MEGATRON_EP}" \
  $@
