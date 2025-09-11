set -x

# Colocated GRPO training+generation for Qwen3-8B on the SWE-Bench task.
# Uses 1 node with 8 GPUs.
# uv run --isolated examples/mini_swe_agent/preprocess_swegym.py --output_dir ~/data/swe_gym_subset
# bash examples/mini_swe_agent/run_mini_swe_8B.sh

DATA_DIR="$HOME/data/swe_gym_subset"
CKPT_PATH="$HOME/ckpts/llm_mini_swe"

# Save trajectories here for debugging
# NOTE: For a multi-node cluster, ensure that this is on NFS so that you can save all trajectories in the same path
MINISWE_TRAJ_DIR="$HOME/mini_swe_agent_trajs"

NUM_GPUS=8
NNODES=1
NUM_INFERENCE_ENGINES=4
TP_SIZE=2
LOGGER=wandb

# We use a small batch size here for demonstration
# NOTE (sumanthrh): The `generator.max_turns` here is actually unused, and we use the `step_limit` from the `swebench.yaml` file. 
# This simply has to be a value > 1
uv run --isolated --extra vllm --extra miniswe --env-file examples/mini_swe_agent/.env.miniswe -m examples.mini_swe_agent.main_mini_swe \
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
  trainer.eval_batch_size=50 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
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
  trainer.project_name="mini_swe" \
  trainer.run_name="mini_swe_8B_swe_gym" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  +generator.miniswe_config_path="examples/mini_swe_agent/swebench.yaml" \
  +generator.miniswe_traj_dir=$MINISWE_TRAJ_DIR
  $@
