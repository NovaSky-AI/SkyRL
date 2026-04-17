set -x

# follow the instructions in examples/train/search/README.md for setting up the
# SearchR1 dataset and starting the local search server.
#
# R3 requires:
#   * trainer.strategy=megatron (only backend that supports routing replay)
#   * generator.inference_engine.distributed_executor_backend=mp
#       - hangs related to Ray Compiled Graph occur with the default ray
#         executor when vLLM returns routed_experts; see
#         https://github.com/vllm-project/vllm/issues/36237
#       - mp executor currently forces single-node serving per engine, which
#         is why INFERENCE_ENGINE_TP is bounded by NUM_GPUS_PER_NODE.
#   * an MoE base model (defaults to Qwen3-30B-A3B).
#
# Multi-turn search-r1 runs up to 4 turns, the generator merges R3
# expert indices across turns using append-only. 
#
# Usage:
#   export WANDB_API_KEY=<your_key_here>
#   bash examples/train/search/run_search_router_replay.sh
#
# Configurable knobs (override via env vars or command-line args):
#   USE_CONVERSATION_MULTI_TURN - "true" to use conversation multi-turn format
#                                  (default: false, matches run_search.sh).
#                                  STEP_WISE is unsupported with R3 and will
#                                  error out in _validate_cfg if enabled.

# Dataset path (shared storage across nodes, or local storage on each node)
DATA_DIR="$HOME/data/searchR1"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B}"

NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

MEGATRON_TP=${MEGATRON_TP:-2}
MEGATRON_PP=${MEGATRON_PP:-1}
MEGATRON_CP=${MEGATRON_CP:-1}
MEGATRON_EP=${MEGATRON_EP:-8}
MEGATRON_ETP=${MEGATRON_ETP:-1}

MICRO_TRAIN_BATCH_SIZE_PER_GPU=${MICRO_TRAIN_BATCH_SIZE_PER_GPU:-1}
MICRO_FORWARD_BATCH_SIZE_PER_GPU=${MICRO_FORWARD_BATCH_SIZE_PER_GPU:-2}

# mp executor requires single-node-per-engine; keep INFERENCE_ENGINE_TP <= NUM_GPUS_PER_NODE.
NUM_INFERENCE_ENGINES=${NUM_INFERENCE_ENGINES:-4}
INFERENCE_ENGINE_TP=${INFERENCE_ENGINE_TP:-8}

# Router replay (R3) settings
ROUTER_REPLAY=${ROUTER_REPLAY:-true}
DISTRIBUTED_EXECUTOR_BACKEND="mp"

# Multi-turn knobs (mirrored from run_search.sh; step-wise disallowed with R3)
: "${USE_CONVERSATION_MULTI_TURN:=false}"

MULTI_TURN_ARGS=""
if [ "$USE_CONVERSATION_MULTI_TURN" = "true" ]; then
  MULTI_TURN_ARGS="generator.use_conversation_multi_turn=true generator.append_eos_token_after_stop_str_in_multi_turn=true"
else
  MULTI_TURN_ARGS="generator.use_conversation_multi_turn=false"
fi

RUN_NAME="skyrl-search_r3_${MODEL_NAME//\//_}_tp${MEGATRON_TP}_pp${MEGATRON_PP}_ep${MEGATRON_EP}"

SKYRL_RAY_PG_TIMEOUT_IN_S=300 uv run --isolated --frozen --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=94 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.moe_enable_routing_replay=$ROUTER_REPLAY \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.enable_return_routed_experts=$ROUTER_REPLAY \
  generator.inference_engine.distributed_executor_backend=$DISTRIBUTED_EXECUTOR_BACKEND \
  trainer.use_sample_packing=true \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BATCH_SIZE_PER_GPU \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE_PER_GPU \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=500 \
  generator.batched=false \
  $MULTI_TURN_ARGS \
  generator.n_samples_per_prompt=5 \
  generator.max_turns=4 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</search>", "</answer>"]' \
  environment.env_class="search" \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.search.log_requests=false \
  environment.skyrl_gym.search.search_url="http://127.0.0.1:8000/retrieve" \
  environment.skyrl_gym.search.topk=3 \
  trainer.logger="wandb" \
  trainer.project_name="skyrl-search-r3" \
  trainer.run_name="${RUN_NAME}" \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/${RUN_NAME}" \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  generator.eval_sampling_params.max_generate_length=500 \
  trainer.export_path="$HOME/${RUN_NAME}/exports" \
  trainer.eval_interval=50 \
  $@
