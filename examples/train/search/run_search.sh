set -x

# Colocated GRPO training+generation for Qwen2.5-3B-Instruct on SearchR1 data.
# Follow the instructions in docs/content/docs/recipes/searchr1.mdx for setup.
#
# Usage:
#   export WANDB_API_KEY=<your_key_here>
#   bash examples/train/search/run_search.sh
#
# Configurable knobs (override via env vars or command-line args):
#   USE_CONVERSATION_MULTI_TURN - set to "true" to use conversation multi-turn format (default: false)
#     When true, also enables append_eos_token_after_stop_str_in_multi_turn=true so that
#     each turn's response ends with the model's EOS token (required for correct behavior
#     when stop strings like </search> or </answer> terminate generation instead of EOS).
#   STEP_WISE - set to "true" to enable step-wise training (default: false)
#     Requires USE_CONVERSATION_MULTI_TURN=true.
#
# Examples:
#   # Default (non-conversation, non-step-wise):
#   bash examples/train/search/run_search.sh
#
#   # Conversation multi-turn format:
#   USE_CONVERSATION_MULTI_TURN=true bash examples/train/search/run_search.sh
#
#   # Step-wise with conversation multi-turn:
#   USE_CONVERSATION_MULTI_TURN=true STEP_WISE=true bash examples/train/search/run_search.sh
#
#   # Override any config via positional args (passed to Hydra):
#   bash examples/train/search/run_search.sh trainer.epochs=2 trainer.eval_interval=10

# path for dataset (.parquet files) containing the prompts and metadata for each question
DATA_DIR="$HOME/data/searchR1"

RUN_NAME="skyrl-search_4turns_maxgeneratelen_500-multiturn-sync-TIS_2.0"

TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

# Configurable knobs with defaults
: "${USE_CONVERSATION_MULTI_TURN:=false}"
: "${STEP_WISE:=false}"

# Build conditional args
MULTI_TURN_ARGS=""
if [ "$USE_CONVERSATION_MULTI_TURN" = "true" ]; then
  MULTI_TURN_ARGS="generator.use_conversation_multi_turn=true generator.append_eos_token_after_stop_str_in_multi_turn=true"
else
  MULTI_TURN_ARGS="generator.use_conversation_multi_turn=false"
fi

: "${MERGE_STEPWISE:=false}"

STEP_WISE_ARGS=""
if [ "$STEP_WISE" = "true" ]; then
  STEP_WISE_ARGS="generator.step_wise_trajectories=true"
  # Step-wise requires conversation multi-turn
  if [ "$USE_CONVERSATION_MULTI_TURN" != "true" ]; then
    echo "WARNING: STEP_WISE=true requires USE_CONVERSATION_MULTI_TURN=true. Enabling it automatically."
    MULTI_TURN_ARGS="generator.use_conversation_multi_turn=true generator.append_eos_token_after_stop_str_in_multi_turn=true"
  fi
  if [ "$MERGE_STEPWISE" = "true" ]; then
    STEP_WISE_ARGS="$STEP_WISE_ARGS generator.merge_stepwise_output=true"
  fi
fi

uv run --isolated --frozen --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=94 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.policy.model.path="Qwen/Qwen2.5-3B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.inference_engine.num_engines=4 \
  generator.inference_engine.tensor_parallel_size=2 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=500 \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  $MULTI_TURN_ARGS \
  $STEP_WISE_ARGS \
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
  trainer.project_name="skyrl-search" \
  trainer.run_name="${RUN_NAME}" \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/${RUN_NAME}" \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  generator.eval_sampling_params.max_generate_length=500 \
  trainer.export_path="$HOME/${RUN_NAME}/exports" \
  trainer.eval_interval=50 \
  $@
