#!/usr/bin/env bash
set -euo pipefail
# Single TP=8 Nemotron engine, real Harbor trials, simulated training only.
# Run from the repository root. Account usage must be checked before launch.
: "${ENV_FILE:=$HOME/hackskyrl/.env.srh}"
: "${DATA_DIR:=/mnt/local_storage/data/harbor/CodeContests-2k}"
: "${MODEL:=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}"
: "${SERVED_MODEL_NAME:=nemotron-550b}"
: "${RUN_NAME:=nemotron-baseline-$(date -u +%m%d-%H%M%S)}"
: "${ARTIFACT_DIR:=$HOME/hackskyrl/logs/0930/nemotron_bench/$RUN_NAME}"
: "${NUM_PARALLEL_GENERATION_WORKERS:=8}"
: "${MAX_CONCURRENCY:=128}"
: "${TRAJECTORIES_PER_SECOND:=null}"
: "${MAX_NUM_SEQS:=32}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_GENERATE_LENGTH:=12288}"
# Long multi-turn trajectories can exceed two hours, especially during output-limit recovery.
: "${AGENT_TIMEOUT_SECONDS:=21600}"
: "${LLM_TIMEOUT_SECONDS:=7200}"
: "${DAYTONA_AUTO_STOP_MINS:=360}"
: "${OFFLOAD:=0}"
: "${CPU_OFFLOAD_BYTES:=549755813888}"
: "${GRAFANA_URL:=http://localhost:9481}"
: "${RAY_VERSION:=2.58.0}"
# The account-wide cap is 500; this local cap leaves room for existing/lingering sandboxes.
if (( MAX_CONCURRENCY < 1 || MAX_CONCURRENCY > 256 )); then
  echo 'MAX_CONCURRENCY must be 1..256; check account headroom separately.' >&2
  exit 2
fi
if (( NUM_PARALLEL_GENERATION_WORKERS < 8 || NUM_PARALLEL_GENERATION_WORKERS > 40 )); then
  echo 'Workers must be 8..40 for minibatch 8 and fixed staleness 4.' >&2
  exit 2
fi
mkdir -p "$ARTIFACT_DIR"
args=(
  "data.train_data=['$DATA_DIR']" "data.val_data=['$DATA_DIR']"
  "harbor_trial_config.trials_dir=$ARTIFACT_DIR/trials"
  "trainer.policy.model.path=$MODEL"
  "generator.inference_engine.served_model_name=$SERVED_MODEL_NAME"
  trainer.fully_async.enabled=true trainer.fully_async.simulate_training=true
  trainer.fully_async.simulate_training_step_seconds=5
  trainer.fully_async.simulate_weight_sync_seconds=0 trainer.fully_async.max_staleness_steps=4
  "trainer.fully_async.num_parallel_generation_workers=$NUM_PARALLEL_GENERATION_WORKERS"
  trainer.algorithm.policy_loss_type=rollout_is trainer.algorithm.advantage_estimator=grpo
  trainer.algorithm.loss_reduction=token_mean "trainer.algorithm.max_seq_len=$MAX_MODEL_LEN"
  trainer.strategy=fsdp trainer.placement.colocate_all=false
  trainer.epochs=1 trainer.max_training_steps=8 trainer.train_batch_size=8 trainer.policy_mini_batch_size=8
  trainer.micro_forward_batch_size_per_gpu=1 trainer.micro_train_batch_size_per_gpu=1
  trainer.eval_before_train=false trainer.eval_interval=0 trainer.ckpt_interval=-1 trainer.hf_save_interval=-1
  trainer.resume_mode=none "trainer.ckpt_path=$ARTIFACT_DIR/ckpts" "trainer.log_path=$ARTIFACT_DIR/infra"
  "generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH"
  generator.inference_engine.backend=vllm generator.inference_engine.num_engines=1
  generator.inference_engine.tensor_parallel_size=8 generator.inference_engine.run_engines_locally=true
  generator.inference_engine.weight_sync_backend=nccl generator.inference_engine.gpu_memory_utilization=0.90
  "generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN"
  generator.inference_engine.engine_init_kwargs.reasoning_parser=nemotron_v3
  generator.inference_engine.engine_init_kwargs.mamba_ssm_cache_dtype=float32
  "generator.inference_engine.max_num_seqs=$MAX_NUM_SEQS"
  "generator.inference_engine.engine_init_kwargs.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  generator.inference_engine.enable_ray_prometheus_stats=true
  generator.inference_engine.router_init_kwargs.request_timeout_secs=7200
  generator.batched=false generator.step_wise_trajectories=true generator.merge_stepwise_output=true
  generator.n_samples_per_prompt=8 generator.rate_limit.enabled=true
  "generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND"
  "generator.rate_limit.max_concurrency=$MAX_CONCURRENCY"
  "harbor_trial_config.agent.kwargs.model_info.max_input_tokens=$MAX_MODEL_LEN"
  "harbor_trial_config.agent.kwargs.model_info.max_output_tokens=$MAX_GENERATE_LENGTH"
  "harbor_trial_config.agent.kwargs.llm_kwargs.max_tokens=$MAX_GENERATE_LENGTH"
  "harbor_trial_config.agent.override_timeout_sec=$AGENT_TIMEOUT_SECONDS"
  "harbor_trial_config.agent.kwargs.llm_kwargs.timeout=$LLM_TIMEOUT_SECONDS"
  harbor_trial_config.agent.kwargs.llm_kwargs.max_retries=0
  "harbor_trial_config.environment.kwargs.auto_stop_interval_mins=$DAYTONA_AUTO_STOP_MINS"
  harbor_trial_config.environment.kwargs.connection_pool_maxsize=null
  trainer.logger=wandb trainer.project_name=nemotron-bench "trainer.run_name=$RUN_NAME"
  trainer.grafana_annotations.enabled=true "trainer.grafana_annotations.url=$GRAFANA_URL"
  trainer.grafana_annotations.organization_id=1
  "trainer.grafana_annotations.record_directory=$ARTIFACT_DIR/annotations"
)
if [[ "$OFFLOAD" == 1 ]]; then
  args+=("generator.inference_engine.engine_init_kwargs.kv_transfer_config={kv_connector: OffloadingConnector, kv_role: kv_both, kv_connector_extra_config: {cpu_bytes_to_use: $CPU_OFFLOAD_BYTES}}")
fi
# CLI overrides appear last: permit clearly labeled smoke/engine-probe exceptions.
printf '%s\n' "${args[@]}" "$@" > "$ARTIFACT_DIR/cli-overrides.txt"
uv_args=()
[[ -z "$RAY_VERSION" ]] || uv_args+=(--with "ray==$RAY_VERSION")
uv run --isolated --frozen --extra fsdp --extra harbor --env-file "$ENV_FILE" "${uv_args[@]}" \
  -m examples.train_integrations.harbor.entrypoints.main_harbor_fully_async_sim \
  "${args[@]}" "$@" 2>&1 | tee "$ARTIFACT_DIR/driver.log"
