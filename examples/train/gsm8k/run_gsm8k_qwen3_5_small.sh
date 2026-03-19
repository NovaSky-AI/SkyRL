set -x

# Small-scale Qwen3.5-0.8B GSM8K run for quick testing.
# Uses 1 GPU, small batches, console logging (no wandb).
#
# Prerequisites:
#   uv run python examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   uv run python scripts/check_fast_path.py  # verify fast path enabled
#
# Usage:
#   bash examples/train/gsm8k/run_gsm8k_qwen3_5_small.sh
#   # Override GPU count: NUM_GPUS=2 bash examples/train/gsm8k/run_gsm8k_qwen3_5_small.sh

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=console}"
: "${MODEL:=Qwen/Qwen3.5-0.8B}"

uv run --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  trainer.epochs=2 \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.ckpt_interval=100 \
  trainer.max_prompt_length=256 \
  generator.sampling_params.max_generate_length=512 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k-qwen3.5-small" \
  trainer.run_name="gsm8k_qwen3.5_0.8B_test" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.ckpt_path="/tmp/skyrl-ckpts/gsm8k_qwen3.5_0.8B_test" \
  $@
