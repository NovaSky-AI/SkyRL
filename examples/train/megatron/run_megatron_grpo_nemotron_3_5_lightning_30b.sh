set -x

# Colocated GRPO training+generation for NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
# on GSM8K with Megatron.
#
# Nemotron-3.5-Lightning is a NemotronH hybrid model: 52 layers interleaving Mamba2,
# attention and MoE blocks (pattern MEMEM*EMEMEM*..., 128 routed experts, 6 active
# per token, 1 shared expert, ~3B active parameters) with relu^2 MLPs.
#
# Two things distinguish it from Nemotron-3-Nano-30B-A3B:
#   * Its HF config declares the layer pattern as `layers_block_type` /
#     `mtp_layers_block_type` lists rather than `hybrid_override_pattern` strings.
#     transformers' NemotronHConfig exposes the string form as a derived property,
#     which is what megatron-bridge and vLLM read, so no conversion is needed.
#   * It ships one MTP head (`num_nextn_predict_layers=1`). RL training does not use
#     it: MegatronWorker drops the head (trainer.mtp.enabled defaults to false) and
#     vLLM skips the `mtp.*` weights, so it never crosses weight sync.
#
# Runs on 1 node of 4 GPUs (Megatron TP=4 EP=4 ETP=1, one colocated vLLM engine at
# TP=4). This is the mesh exercised by
# tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py
# (id: nemotron3.5-lightning_tp4_ep4_h100). To scale to 8 GPUs, keep TP=4 and raise
# EP to 8 (128 experts / 8 = 16 per GPU), as in run_megatron_grpo_glm4_7_30b.sh.
#
# Setup:
#   1. Install deps:
#        uv sync --extra megatron
#   2. Prepare data:
#        uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   3. Run:
#        export WANDB_API_KEY=<your_key_here>  # or set LOGGER=console below
#        bash examples/train/megatron/run_megatron_grpo_nemotron_3_5_lightning_30b.sh

MODEL_NAME="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
DATA_DIR=${DATA_DIR:-"$HOME/data/gsm8k"}
CKPT_DIR=${CKPT_DIR:-"$HOME/ckpts/nemotron_3_5_lightning_30b_a3b_grpo_megatron"}
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"

NUM_NODES=1
NUM_GPUS=4

# Megatron parallelism: TP=4 shards the dense/attention layers, EP=4 spreads the
# 128 routed experts (32 per GPU). TP=1 OOMs in the EP alltoall because the dense
# layers are then replicated on every GPU.
MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=4
MEGATRON_ETP=1

# vLLM inference: 1 engine x TP=4, colocated on the same 4 GPUs.
# Nemotron-3.5-Lightning defaults to a 262k context, which would size the KV pool
# far past what is left next to the Megatron policy shard -- cap it.
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=4
INFERENCE_ENGINE_MAX_MODEL_LEN=4096

# NemotronH attention is standard GQA (32 heads / 2 KV heads), so flash attention works.
FLASH_ATTN=true

# MoE routing flags (sigmoid scoring with expert bias, matching the checkpoint)
MOE_TOKEN_DISPATCHER="alltoall"
MOE_ROUTER_LB="none"
MOE_GROUPED_GEMM=true
MOE_ROUTER_SCORE_FN="sigmoid"
MOE_ROUTER_EXPERT_BIAS=true

# CPU optimizer offload to fit in 80GB GPUs. Megatron's DistributedOptimizer keeps
# the fp32 master weights + AdamW moments resident, which does not fit next to the
# colocated vLLM pool on 80GB cards. Set to false if you have the headroom.
OPTIMIZER_CPU_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

ENFORCE_EAGER=false

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
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.inference_engine.engine_init_kwargs.max_model_len=$INFERENCE_ENGINE_MAX_MODEL_LEN \
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
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.remove_microbatch_padding=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.4 \
  trainer.logger="$LOGGER" \
  trainer.project_name="nemotron_3_5_lightning_30b_grpo" \
  trainer.run_name="nemotron_3_5_lightning_30b_a3b_grpo_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_DIR" \
  $@
