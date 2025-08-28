set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-1.5B-Instruct on GSM8k dataset.
# Uses 1 node with 4 GPUs.
# uv run examples/llm_as_a_judge/gsm8k_dataset_judge.py --output_dir $HOME/data/gsm8k_llm_judge
# add OPENAI_API_KEY and WANDB_API_KEY to .env.llm_judge
# bash examples/llm_as_a_judge/run_llm_judge.sh

DATA_DIR="/mnt/local_storage/skyrl_v0_293/"
CKPT_PATH="$HOME/ckpts/llm_mini_swe"

NUM_GPUS=2
NUM_INFERENCE_ENGINES=1
TP_SIZE=1
LOGGER=wandb

# We use a smaller batch size here for demonstration
uv run --isolated --extra vllm --extra miniswe --env-file examples/mini_swe_agent/.env.miniswe -m examples.mini_swe_agent.main_mini_swe \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.placement.colocate_all=false \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=40 \
  trainer.micro_train_batch_size_per_gpu=40 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=False \
  generator.remote_inference_engine_urls="['127.0.0.1:8001']" \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="mini_swe" \
  trainer.run_name="gsm8k_mini_swe" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_mini_swe_1.5B_ckpt" \
  +generator.miniswe_config_path="/home/ray/default/SkyRL/skyrl-train/examples/mini_swe_agent/swebench.yaml" \
  +generator.miniswe_traj_dir="/mnt/local_storage/mini_swe_agent"
  $@