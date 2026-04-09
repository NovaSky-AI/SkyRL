set -x

# Multi-paper RLM eval-only: generate rollouts and report metrics (no training).
#
# 1. Create data: uv run -- python examples/train/rlm/multi_paper_dataset.py --output_dir $DATA_DIR
# 2. Run: bash examples/train/rlm/run_multi_paper_eval.sh

: "${UV_CACHE_DIR:=/workspace/.uv-cache}"
: "${UV_PROJECT_ENVIRONMENT:=/workspace/SkyRL/.venv}"
export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT

: "${DATA_DIR:=$HOME/data/multi-paper}"
: "${NUM_ENGINES:=2}"
: "${TP_SIZE:=4}"
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MODEL_PATH:=alphaXiv/rlm-sft-multi-9b-step-500}"
: "${ROLLOUT_OUTPUT_DIR:=$(pwd)/tmp/multi-paper-eval/rollouts}"

uv run --extra fsdp -m skyrl.train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=rlm \
  generator.step_wise_trajectories=true \
  generator.max_turns=10 \
  generator.batched=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=false \
  trainer.max_prompt_length=32768 \
  generator.max_input_length=32768 \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.eval_sampling_params.max_generate_length=4096 \
  generator.eval_sampling_params.temperature=0.7 \
  generator.eval_sampling_params.top_p=0.8 \
  generator.eval_sampling_params.top_k=20 \
  generator.eval_sampling_params.min_p=0.0 \
  generator.eval_sampling_params.repetition_penalty=1.0 \
  generator.eval_sampling_params.additional_kwargs.presence_penalty=1.5 \
  generator.eval_n_samples_per_prompt=1 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  trainer.eval_batch_size=8 \
  trainer.dump_eval_results=true \
  trainer.export_path="$(pwd)/tmp/multi-paper-eval" \
  trainer.logger="$LOGGER" \
  trainer.project_name="rlm" \
  trainer.run_name="multi_paper_eval" \
  environment.skyrl_gym.rlm.custom_system_prompt=multipaper \
  environment.skyrl_gym.rlm.child_system_prompt=multipaper_child \
  generator.child_openrouter_model="openai/gpt-5.4-nano" \
  environment.skyrl_gym.rlm.rollout_output_dir="$ROLLOUT_OUTPUT_DIR" \
  "$@"
