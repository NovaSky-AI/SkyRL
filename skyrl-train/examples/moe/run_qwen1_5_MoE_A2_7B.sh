set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/gsm8k/run_gsm8k.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=4
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"
# INFERENCE_BACKEND="sglang"

# 4/2/2 doesn't work
# 980s for forward/values/logprobs, then get an error 
    # torch.utils.checkpoint.CheckpointError: torch.utils.checkpoint: A different number of tensors was saved during the original forward and recomputation.
    # Number of tensors saved during forward: 1010
    # Number of tensors saved during recomputation: 674
# 

# Remote is failing on weight sync ??

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --isolated --extra $INFERENCE_BACKEND --env-file .env -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.inference_engine_expert_parallel_size=1 \
  generator.inference_engine_data_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=40 \
  trainer.micro_train_batch_size_per_gpu=40 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="moe" \
  trainer.run_name="moe_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/moe_1.5B_ckpt" \
  $@