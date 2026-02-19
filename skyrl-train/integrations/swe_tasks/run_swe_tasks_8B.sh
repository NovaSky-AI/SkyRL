set -x

# GRPO training for Qwen3-8B on swe-task-generator tasks.
# Uses 1 node with 8 GPUs (8xH100 recommended).
#
# Prerequisites:
#   1. Prepare dataset:
#      uv run --isolated integrations/swe_tasks/prepare_dataset.py \
#        --tasks-dir /path/to/swe-task-generator/tasks \
#        --output-dir ~/data/swe_tasks
#
#   2. Ensure Docker images are accessible:
#      docker pull erranli/swe-task-marshmallow-code-marshmallow-2894:latest
#      docker pull erranli/swe-task-marshmallow-code-marshmallow-2892:latest
#
#   3. Run:
#      cd SkyRL/skyrl-train
#      bash integrations/swe_tasks/run_swe_tasks_8B.sh

DATA_DIR="${DATA_DIR:-$HOME/data/swe_tasks}"
CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/swe_tasks}"
SWE_TASKS_TRAJ_DIR="${SWE_TASKS_TRAJ_DIR:-$HOME/swe_tasks_trajs}"

NUM_GPUS=8
NNODES=1
NUM_INFERENCE_ENGINES=4
TP_SIZE=2
LOGGER=wandb

uv run --isolated --extra vllm --extra miniswe \
  -m integrations.swe_tasks.entrypoints.main_swe_tasks \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=2 \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=2 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=30720 \
  generator.max_turns=20 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host='127.0.0.1' \
  generator.http_endpoint_port=8001 \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="swe_tasks" \
  trainer.run_name="swe_tasks_8B_marshmallow" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  +generator.swe_tasks_config_path="integrations/swe_tasks/swe_tasks.yaml" \
  +generator.swe_tasks_traj_dir=$SWE_TASKS_TRAJ_DIR \
  $@
