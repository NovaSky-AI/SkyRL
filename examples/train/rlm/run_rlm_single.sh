set -x

# Single-paper RLM training.
# The top-level agent runs as a child-style evidence extractor (no parent orchestration layer).
#
# 1. Create data: uv run -- python examples/train/rlm/rlm_dataset_synthetic_single.py --output_dir $DATA_DIR
# 2. Run: bash examples/train/rlm/run_rlm_single.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
: "${UV_CACHE_DIR:=$PROJECT_ROOT/.uv-cache}"
: "${UV_PROJECT_ENVIRONMENT:=$PROJECT_ROOT/.venv}"
export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT

: "${DATA_DIR:=$HOME/data/rlm-synthetic}"
: "${NUM_ENGINES:=2}"
: "${TP_SIZE:=4}"
: "${TRAIN_GPUS:=8}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${JUDGE_MODEL:=gpt-5.4-mini-2026-03-17}"

export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"

uv run --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=rlm \
  generator.step_wise_trajectories=true \
  generator.enable_child_agents=false \
  generator.max_turns=10 \
  generator.batched=false \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="alphaXiv/evidence-rlm-sft-4b" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$TRAIN_GPUS \
  trainer.placement.ref_num_gpus_per_node=$TRAIN_GPUS \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.epochs=5 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.eval_batch_size=16 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=100 \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=32768 \
  generator.sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=0.7 \
  generator.sampling_params.top_p=0.8 \
  generator.sampling_params.top_k=20 \
  generator.sampling_params.min_p=0.0 \
  generator.sampling_params.repetition_penalty=1.0 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.01 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.max_input_length=32768 \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  generator.inference_engine.enforce_eager=false \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.n_samples_per_prompt=8 \
  trainer.logger="['console','wandb']" \
  trainer.project_name="rlm" \
  trainer.run_name="rlm_single_paper_grpo" \
  trainer.log_path="$(pwd)/.neer/artifacts/skyrl-logs" \
  trainer.ckpt_path="$(pwd)/.neer/artifacts/ckpts/rlm_ckpt" \
  trainer.export_path="$(pwd)/.neer/artifacts/rlm_exports" \
  trainer.dump_eval_results=true \
  environment.skyrl_gym.rlm.rollout_output_dir="$(pwd)/.neer/artifacts/rlm_rollouts" \
  environment.skyrl_gym.rlm.custom_system_prompt=multipaper_child \
  generator.judge_reward_model="$JUDGE_MODEL" \
  "$@"
