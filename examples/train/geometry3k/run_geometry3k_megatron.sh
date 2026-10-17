set -x

# Multi-turn GRPO training for Geometry-3K (VLM), Megatron backend.
#
# Megatron variant of run_geometry3k.sh; all non-backend settings are identical.
# Defaults to TP=2, PP=1 (DP=4 on 8 GPUs). VLMs on Megatron do not support
# microbatch padding removal (sample packing) or context parallelism.
#
# uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir $HOME/data/geometry_3k
# bash examples/train/geometry3k/run_geometry3k_megatron.sh

: "${DATA_DIR:="$HOME/data/geometry_3k"}"
: "${NUM_GPUS:=8}"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "=== Generating Geometry-3K dataset ==="
  uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir "$DATA_DIR"
fi
: "${LOGGER:=console}"
: "${MEGATRON_TP:=2}"
: "${MEGATRON_PP:=1}"
: "${EXPORT_PATH:="$HOME/exports/geometry3k_vlm_megatron"}"
: "${CKPT_PATH:="$HOME/ckpts/geometry3k_vlm_megatron_ckpt"}"

uv run --isolated --extra megatron --with pylatexenc \
  python examples/train/geometry3k/geometry3k_entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-VL-8B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=6 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=5 \
  trainer.remove_microbatch_padding=false \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=2048 \
  generator.max_turns=3 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  generator.vision_language_generator=true \
  environment.env_class=geometry3k \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="geometry3k" \
  trainer.run_name="geometry3k_vlm_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  "$@"
