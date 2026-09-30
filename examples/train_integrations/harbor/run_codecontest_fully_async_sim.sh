#!/usr/bin/env bash
set -euo pipefail

# Real CodeContests sandbox rollouts; training and weight synchronization are simulated.
# Run from the SkyRL repository root. Use --env-file for Daytona/W&B credentials.
# Default: five simulated steps, one inference GPU, at most four Daytona trials.
: "${ENV_FILE:=$HOME/hackskyrl/.env.srh}"
: "${DATA_DIR:=/mnt/local_storage/data/harbor/CodeContests-2k}"
: "${ARTIFACT_DIR:=$HOME/hackskyrl/logs/0929/async-sim-codecontests}"
: "${MODEL:=Qwen/Qwen3-8B}"
: "${SERVED_MODEL_NAME:=Qwen3-8B}"
: "${RUN_NAME:=codecontests-fully-async-sim}"
: "${MINI_BATCH_SIZE:=2}"
: "${NUM_PARALLEL_GENERATION_WORKERS:=2}"
: "${N_SAMPLES_PER_PROMPT:=2}"
: "${MAX_CONCURRENCY:=4}"
: "${MAX_TRAINING_STEPS:=5}"
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_GENERATE_LENGTH:=12288}"
: "${SIM_STEP_SECONDS:=5}"
: "${SIM_WEIGHT_SYNC_SECONDS:=0}"
: "${LOGGER:=wandb}"

mkdir -p "$ARTIFACT_DIR"
CHAT_TEMPLATE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"
uv_args=()
# Optional overlay for a managed Ray cluster whose version differs from uv.lock.
if [[ -n "${RAY_VERSION:-}" ]]; then
  uv_args+=(--with "ray==$RAY_VERSION")
fi

uv run --isolated --frozen --extra fsdp --extra harbor --env-file "$ENV_FILE" "${uv_args[@]}" \
  -m examples.train_integrations.harbor.entrypoints.main_harbor_fully_async_sim \
  "data.train_data=['$DATA_DIR']" "data.val_data=['$DATA_DIR']" \
  "harbor_trial_config.trials_dir=$ARTIFACT_DIR/trials" \
  "trainer.policy.model.path=$MODEL" \
  "generator.inference_engine.served_model_name=$SERVED_MODEL_NAME" \
  trainer.fully_async.enabled=true trainer.fully_async.simulate_training=true \
  "trainer.fully_async.simulate_training_step_seconds=$SIM_STEP_SECONDS" \
  "trainer.fully_async.simulate_weight_sync_seconds=$SIM_WEIGHT_SYNC_SECONDS" \
  trainer.fully_async.max_staleness_steps=1 \
  "trainer.fully_async.num_parallel_generation_workers=$NUM_PARALLEL_GENERATION_WORKERS" \
  trainer.algorithm.policy_loss_type=rollout_is trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=token_mean \
  "trainer.algorithm.max_seq_len=$MAX_MODEL_LEN" \
  trainer.strategy=fsdp trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.epochs=1 "trainer.max_training_steps=$MAX_TRAINING_STEPS" \
  "trainer.train_batch_size=$MINI_BATCH_SIZE" "trainer.policy_mini_batch_size=$MINI_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu=1 trainer.micro_train_batch_size_per_gpu=1 \
  trainer.eval_before_train=false trainer.eval_interval=0 \
  trainer.ckpt_interval=-1 trainer.hf_save_interval=-1 trainer.resume_mode=none \
  "trainer.ckpt_path=$ARTIFACT_DIR/ckpts" "trainer.log_path=$ARTIFACT_DIR/infra" \
  "generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH" \
  generator.inference_engine.backend=vllm generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.run_engines_locally=true generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.35 \
  "generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN" \
  "generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH" \
  generator.inference_engine.enable_ray_prometheus_stats=true \
  generator.batched=false generator.step_wise_trajectories=true generator.merge_stepwise_output=true \
  "generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT" \
  generator.rate_limit.enabled=true generator.rate_limit.trajectories_per_second=2 \
  "generator.rate_limit.max_concurrency=$MAX_CONCURRENCY" \
  "harbor_trial_config.agent.kwargs.model_info.max_input_tokens=$MAX_MODEL_LEN" \
  "harbor_trial_config.agent.kwargs.model_info.max_output_tokens=$MAX_GENERATE_LENGTH" \
  "harbor_trial_config.agent.kwargs.llm_kwargs.max_tokens=$MAX_GENERATE_LENGTH" \
  harbor_trial_config.environment.kwargs.connection_pool_maxsize=null \
  "trainer.logger=$LOGGER" trainer.project_name=harbor-async-sim "trainer.run_name=$RUN_NAME" \
  "$@"
