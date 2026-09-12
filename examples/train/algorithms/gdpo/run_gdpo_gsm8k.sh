set -x

# Colocated GDPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.
#
# GDPO (https://arxiv.org/abs/2601.05242) normalizes each reward component separately within a
# prompt group before summing, so distinct reward combinations keep distinct advantages instead of
# collapsing onto the same value the way a summed reward does under GRPO.
#
# It requires the generator to emit `reward_components` (one score per objective per response), so
# this uses the `gsm8k_multi_reward` env, which scores format and correctness separately.

# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/algorithms/gdpo/run_gdpo_gsm8k.sh


# You can override the default values with e.g.: `NUM_GPUS=1 bash examples/train/algorithms/gdpo/run_gdpo_gsm8k.sh`.

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout
: "${INFERENCE_BACKEND:=vllm}"

# GDPO parameters
: "${ADV_ESTIMATOR:=gdpo}"
# GDPO batch-normalizes the summed per-component advantages itself, so the trainer-level pass stays off.
: "${ADVANTAGE_BATCH_NORMALIZE:=false}"

# Other algorithm parameters
: "${USE_KL_LOSS:=true}"

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="$ADV_ESTIMATOR" \
  trainer.algorithm.advantage_batch_normalize=$ADVANTAGE_BATCH_NORMALIZE \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=gsm8k_multi_reward \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k" \
  trainer.run_name="gdpo_gsm8k" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_ckpt" \
  "$@"
