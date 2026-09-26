set -x

# Multi-turn GRPO training for Geometry-3K (VLM) with Qwen3-VL-30B-A3B (MoE), Megatron backend.
#
# Larger-model variant of run_geometry3k_megatron.sh. Targets 2 nodes x 8 H100 (80GB).
# Exercises the Megatron VLM path with expert parallelism (Qwen3VLMoEBridge in Megatron-Bridge)
# and multi-node colocated vLLM.
#
# Parallelism (override via env):
#   Training : TP=2, PP=1, CP=1, EP=8, ETP=1  (TP=2 keeps Megatron SP at the config verified for VLMs;
#              fall back to MEGATRON_TP=4 MEGATRON_PP=2 if the optimizer state does not fit)
#   Inference: 4 vLLM engines x TP=4 (fall back to NUM_INFERENCE_ENGINES=2 INFERENCE_ENGINE_TP=8)
# VLMs on Megatron: no microbatch padding removal (packing) and no context parallelism.
#
# uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir $HOME/data/geometry_3k
# bash examples/train/geometry3k/run_geometry3k_30b_a3b_megatron.sh

: "${DATA_DIR:="$HOME/data/geometry_3k"}"
: "${NUM_NODES:=2}"
: "${NUM_GPUS:=8}"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "=== Generating Geometry-3K dataset ==="
  uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir "$DATA_DIR"
fi
: "${LOGGER:=console}"
: "${MODEL_NAME:="Qwen/Qwen3-VL-30B-A3B-Instruct"}"
: "${MEGATRON_TP:=2}"
: "${MEGATRON_PP:=1}"
: "${MEGATRON_CP:=1}"
: "${MEGATRON_EP:=8}"
: "${MEGATRON_ETP:=1}"
: "${NUM_INFERENCE_ENGINES:=4}"
: "${INFERENCE_ENGINE_TP:=4}"
: "${RECOMPUTE_GRANULARITY:="full"}"
: "${RECOMPUTE_METHOD:="uniform"}"
: "${RECOMPUTE_NUM_LAYERS:=1}"
: "${TRAIN_BATCH_SIZE:=128}"
: "${MINI_BATCH_SIZE:=64}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_RESPONSE_LENGTH:=4096}"
# Whole-conversation budget checked before every turn; must exceed prompt + max_turns * response
# or retry turns get cut off (VLM_GAPS.md #15).
: "${MAX_INPUT_LENGTH:=8192}"
: "${EPOCHS:=4}"
: "${LR:=1.0e-6}"
: "${CKPT_PATH:="$HOME/ckpts/geometry3k_vlm_30b_a3b_megatron_ckpt"}"
: "${EXPORT_PATH:="$HOME/exports/geometry3k_vlm_30b_a3b_megatron"}"

uv run --isolated --extra megatron --with pylatexenc \
  python examples/train/geometry3k/geometry3k_entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=$RECOMPUTE_GRANULARITY \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=$RECOMPUTE_METHOD \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=$RECOMPUTE_NUM_LAYERS \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.epochs=$EPOCHS \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=10 \
  trainer.remove_microbatch_padding=false \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.max_turns=3 \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  generator.vision_language_generator=true \
  environment.env_class=geometry3k \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="geometry3k" \
  trainer.run_name="geometry3k_vlm_30b_a3b_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_ep${MEGATRON_EP}" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  "$@"
