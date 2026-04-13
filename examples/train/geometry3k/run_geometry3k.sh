set -x

# Multi-turn GRPO training for Geometry-3K (VLM).

# uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir $HOME/data/geometry_3k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/geometry3k/run_geometry3k.sh

: "${DATA_DIR:="$HOME/data/geometry_3k"}"
: "${NUM_GPUS:=8}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${EXPORT_PATH:="$HOME/exports/geometry3k_vlm"}"
: "${DUMP_EVAL_RESULTS:=true}"

_SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra fsdp \
  python examples/train/geometry3k/geometry3k_entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/train-dev.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-VL-8B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
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
  trainer.train_batch_size=256 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=5 \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=2048 \
  generator.max_turns=3 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.is_vlm=true \
  environment.env_class=geometry3k \
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="geometry3k" \
  trainer.run_name="geometry3k_vlm" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results="$DUMP_EVAL_RESULTS" \
  trainer.ckpt_path="$HOME/ckpts/geometry3k_vlm_ckpt" \
  $@
