# Colocated partial-rollout GRPO for Qwen2.5-1.5B-Instruct on GSM8K.
# Same knobs as examples/train/fully_async/fully_async_run_gsm8k.sh except:
# - colocate_all=true: all NUM_GPUS are shared between the inference engines and the trainer
# - NUM_PARALLEL_GENERATION_WORKERS is the number of groups in flight; each step trains on the first
#   MINI_BATCH_SIZE to finish and cuts the rest, so 1 - MINI_BATCH_SIZE/NUM_PARALLEL_GENERATION_WORKERS
#   of the in-flight groups resume under the next policy.
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/fully_async/partial_rollout_run_gsm8k.sh

set -x

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout / or use wandb
: "${INFERENCE_BACKEND:=vllm}"
: "${MINI_BATCH_SIZE:=256}"
# 256 of 341 in-flight groups finish before the cut: 25% of the in-flight groups resume each step.
: "${NUM_PARALLEL_GENERATION_WORKERS:=341}"

SEQUENCE_MASK_METRIC=geometric
GEO_MASK_HIGH=1.01
GEO_MASK_LOW=0.99
ENFORCE_EAGER=false

RUN_NAME=gsm8k-partial-rollout-qwen2.5_1.5B-geoMask${GEO_MASK_LOW}_${GEO_MASK_HIGH}-batch${MINI_BATCH_SIZE}-inflight${NUM_PARALLEL_GENERATION_WORKERS}-${NUM_GPUS}gpus

uv run --isolated --extra fsdp -m examples.train.fully_async.main_fully_async \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.fully_async.enabled=true \
  trainer.fully_async.num_parallel_generation_workers=${NUM_PARALLEL_GENERATION_WORKERS} \
  trainer.fully_async.max_staleness_steps=0 \
  trainer.fully_async.partial_rollout.cut_mode=abort \
  trainer.algorithm.policy_loss_type="rollout_is" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.off_policy_correction.sequence_mask_metric=$SEQUENCE_MASK_METRIC \
  trainer.algorithm.off_policy_correction.geo_mask_high=$GEO_MASK_HIGH \
  trainer.algorithm.off_policy_correction.geo_mask_low=$GEO_MASK_LOW \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  generator.inference_engine.router_init_kwargs.policy=sticky_least_loaded \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=4 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${MINI_BATCH_SIZE} \
  trainer.policy_mini_batch_size=${MINI_BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k-partial-rollout" \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  $@
