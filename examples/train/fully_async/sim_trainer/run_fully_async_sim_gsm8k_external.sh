set -x

# Fully async GRPO training with a simulated trainer for Qwen2.5-1.5B-Instruct on GSM8K.
# Uses an external proxy and server url to benchmark against an existing deployment.
# Each step generates a mini-batch, sleeps SIM_STEP_SECONDS (stand-in for forward/backward), then
# pause/resume — exercising the whole generation-side loop end to end and logging to wandb.
#
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k   # one-time
#   Run:
#     1. Launch external servers with 
#          bash fully-async-sim-trainer/examples/train/fully_async/sim_trainer/run_serve_qwen1.5b.sh
#     2. Launch simulated fully async training - ensure you use the right `EXTERNAL_PROXY_URL` and `EXTERNAL_SERVER_URLS`
#          bash fully-async-sim-trainer/examples/train/fully_async/sim_trainer/run_fully_async_sim_gsm8k_external.sh

: "${EXTERNAL_PROXY_URL:="http://127.0.0.1:40859"}"
: "${EXTERNAL_SERVER_URLS:=['http://127.0.0.1:8000','http://127.0.0.1:8001']}"

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${NUM_INFERENCE_ENGINES:=2}"
: "${LOGGER:=wandb}"

# Tiny fully-async knobs for a quick test.
: "${MINI_BATCH_SIZE:=4}"
: "${MAX_STALENESS_STEPS:=1}"
: "${NUM_PARALLEL_GENERATION_WORKERS:=4}"   # mb <= npgw <= mb*(staleness+1) = 8
: "${SIM_STEP_SECONDS:=5}"
: "${SIM_WEIGHT_SYNC_SECONDS:=0}"
: "${N_SAMPLES_PER_PROMPT:=2}"

RUN_NAME="${RUN_NAME:-gsm8k-sim-qwen0.5b}"

uv run --isolated --extra fsdp \
  -m examples.train.fully_async.main_fully_async_sim \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.fully_async.simulate_training=true \
  trainer.fully_async.simulate_training_step_seconds=$SIM_STEP_SECONDS \
  trainer.fully_async.simulate_weight_sync_seconds=$SIM_WEIGHT_SYNC_SECONDS \
  trainer.fully_async.max_staleness_steps=${MAX_STALENESS_STEPS} \
  trainer.fully_async.num_parallel_generation_workers=${NUM_PARALLEL_GENERATION_WORKERS} \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=false \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.epochs=1 \
  trainer.train_batch_size=${MINI_BATCH_SIZE} \
  trainer.policy_mini_batch_size=${MINI_BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=none \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=512 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.external_proxy_url=$EXTERNAL_PROXY_URL \
  generator.inference_engine.external_server_urls="$EXTERNAL_SERVER_URLS" \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.enforce_eager=true \
  generator.batched=false \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k-sim-test" \
  trainer.run_name=${RUN_NAME} \
  $@
